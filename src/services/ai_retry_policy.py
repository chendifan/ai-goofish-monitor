"""AI 调用重试策略：统一尝试计数、退避节奏与输出预算递增。"""

from dataclasses import dataclass
from typing import Optional

from src.infrastructure.config.settings import AISettings

OUTPUT_BUDGET_MARKERS = (
    "finish_reason=length",
    "structured output was incomplete",
    "output budget",
    "output length",
    "truncat",
    "incomplete",
)
STRUCTURED_OUTPUT_FORMAT_MARKERS = (
    "did not return valid json",
    "invalid json",
    "invalid control character",
    "expecting ',' delimiter",
    "expecting value",
)
OUTPUT_BUDGET_ERROR_TYPES = {"max_tokens", "output_length_exceeded"}
STRUCTURED_OUTPUT_FORMAT_TYPES = {"structured_output_failed", "invalid_json", "json_parse_error"}
RATE_LIMIT_MARKERS = (
    "rate limit",
    "too many requests",
    "429",
)
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "APIReadTimeoutError",
}
TRANSIENT_MESSAGE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
)


def _error_type(error: Exception) -> str:
    return str(getattr(error, "type", "") or "").lower()


def is_output_budget_error(error: Exception) -> bool:
    """识别输出预算不足（思考过程或正文被截断）的错误。"""
    message = str(error).lower()
    if any(marker in message for marker in OUTPUT_BUDGET_MARKERS):
        return True

    return _error_type(error) in OUTPUT_BUDGET_ERROR_TYPES


def is_structured_output_format_error(error: Exception) -> bool:
    """模型返回的 JSON 本身不合法（坏 JSON、空内容），加大预算解决不了这类问题。"""
    if is_output_budget_error(error):
        return False
    message = str(error).lower()
    if any(marker in message for marker in STRUCTURED_OUTPUT_FORMAT_MARKERS):
        return True
    return _error_type(error) in STRUCTURED_OUTPUT_FORMAT_TYPES


def is_retryable_error(error: Exception) -> bool:
    """值得再试一次的情况，包括截断、坏 JSON 与瞬时故障。"""
    return (
        is_transient_error(error)
        or is_output_budget_error(error)
        or is_structured_output_format_error(error)
    )


def is_transient_error(error: Exception) -> bool:
    """识别重试即可恢复的瞬时错误（超时、连接、限流、5xx）。"""
    if is_output_budget_error(error):
        return False

    status_code = getattr(error, "status_code", None)
    if status_code in TRANSIENT_STATUS_CODES:
        return True

    if type(error).__name__ in TRANSIENT_ERROR_NAMES:
        return True

    message = str(error).lower()
    return any(marker in message for marker in RATE_LIMIT_MARKERS + TRANSIENT_MESSAGE_MARKERS)


def retry_after_seconds(error: Exception, cap: float) -> Optional[float]:
    """读取网关返回的 Retry-After，超限时按 cap 收敛。"""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    value = None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        value = None

    if value is None:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            value = body.get("retry_after") or body.get("retry_after_seconds")

    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(seconds, cap))


@dataclass(frozen=True)
class AIRetryPolicy:
    """AI 调用重试策略（应用层是唯一重试主体，SDK 内部重试应设为 0）。"""

    max_attempts: int = 4
    base_output_tokens: int = 32000
    output_token_cap: int = 64000
    base_delay_seconds: float = 3.0
    delay_cap_seconds: float = 30.0

    @classmethod
    def from_settings(cls, settings: Optional[AISettings] = None) -> "AIRetryPolicy":
        settings = settings or AISettings()
        return cls(
            max_attempts=max(1, settings.max_attempts),
            base_output_tokens=max(1, settings.max_output_tokens),
            output_token_cap=max(1, settings.max_output_tokens_cap),
            base_delay_seconds=max(0.0, settings.retry_base_delay_seconds),
            delay_cap_seconds=max(1.0, settings.retry_delay_cap_seconds),
        )

    def attempts(self) -> range:
        return range(1, self.max_attempts + 1)

    def describe_attempt(self, attempt: int) -> str:
        return f"第 {attempt}/{self.max_attempts} 次尝试"

    def has_next_attempt(self, attempt: int) -> bool:
        return attempt < self.max_attempts

    def next_attempt_number(self, attempt: int) -> int:
        return min(attempt + 1, self.max_attempts)

    def escalate_output_tokens(self, current: int) -> int:
        """预算不足时按倍递增，直到上限。"""
        return min(max(current, current * 2), self.output_token_cap)

    def delay_seconds(self, attempt: int, error: Optional[Exception] = None) -> float:
        retry_after = retry_after_seconds(error, self.delay_cap_seconds)
        if retry_after is not None:
            return retry_after
        return min(self.delay_cap_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))


def format_retry_note(policy: AIRetryPolicy, attempt: int, delay: float) -> str:
    next_attempt = policy.next_attempt_number(attempt)
    if delay > 0:
        return f"等待 {delay:.1f}s 后进行第 {next_attempt}/{policy.max_attempts} 次尝试"
    return f"准备第 {next_attempt}/{policy.max_attempts} 次尝试"
