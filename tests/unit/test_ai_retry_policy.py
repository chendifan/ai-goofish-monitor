from types import SimpleNamespace

from src.services.ai_retry_policy import (
    AIRetryPolicy,
    format_retry_note,
    is_output_budget_error,
    is_transient_error,
    retry_after_seconds,
)

BUDGET_ERROR = Exception(
    "Error code: 502 - {'error': {'type': 'structured_output_failed', 'code': "
    "'structured_output_failed', 'message': 'structured output was incomplete "
    "(finish_reason=length); increase the output budget'}}"
)


def test_is_output_budget_error_detects_structured_output_failure():
    assert is_output_budget_error(BUDGET_ERROR) is True


def test_is_output_budget_error_ignores_plain_timeout():
    assert is_output_budget_error(Exception("Request timed out.")) is False


def test_is_transient_error_detects_timeout_and_rate_limit():
    assert is_transient_error(Exception("Request timed out.")) is True
    assert is_transient_error(Exception("Error code: 429 - too many requests")) is True


def test_is_transient_error_separates_budget_errors_from_network_errors():
    assert is_transient_error(BUDGET_ERROR) is False


def test_retry_after_header_overrides_exponential_backoff():
    error = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after": "2"}),
        body=None,
    )
    policy = AIRetryPolicy(base_delay_seconds=3.0, delay_cap_seconds=30.0)
    assert retry_after_seconds(error, policy.delay_cap_seconds) == 2.0
    assert policy.delay_seconds(1, error) == 2.0


def test_policy_escalates_output_budget_until_cap():
    policy = AIRetryPolicy(base_output_tokens=4000, output_token_cap=16000)
    assert policy.escalate_output_tokens(4000) == 8000
    assert policy.escalate_output_tokens(8000) == 16000
    assert policy.escalate_output_tokens(16000) == 16000


def test_policy_delay_grows_exponentially_and_is_capped():
    policy = AIRetryPolicy(max_attempts=4, base_delay_seconds=3.0, delay_cap_seconds=10.0)
    assert policy.delay_seconds(1) == 3.0
    assert policy.delay_seconds(2) == 6.0
    assert policy.delay_seconds(3) == 10.0
    assert policy.delay_seconds(4) == 10.0


def test_attempt_numbers_start_at_one_and_accumulate():
    policy = AIRetryPolicy(max_attempts=3)
    assert list(policy.attempts()) == [1, 2, 3]
    assert policy.describe_attempt(1) == "第 1/3 次尝试"
    assert policy.describe_attempt(2) == "第 2/3 次尝试"
    assert policy.next_attempt_number(2) == 3
    assert policy.has_next_attempt(3) is False


def test_format_retry_note_uses_zero_delay_when_backoff_is_disabled():
    policy = AIRetryPolicy(max_attempts=4, base_delay_seconds=0.0)
    assert format_retry_note(policy, 1, 0.0) == "准备第 2/4 次尝试"
    assert format_retry_note(policy, 1, 2.5) == "等待 2.5s 后进行第 2/4 次尝试"


def test_policy_reads_settings_from_env(monkeypatch):
    monkeypatch.setenv("AI_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "6000")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS_CAP", "12000")
    policy = AIRetryPolicy.from_settings()
    assert policy.max_attempts == 5
    assert policy.base_output_tokens == 6000
    assert policy.output_token_cap == 12000
