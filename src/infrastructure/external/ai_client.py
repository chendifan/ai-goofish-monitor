"""
AI 客户端封装
提供统一的 AI 调用接口
"""
import asyncio
import ipaddress
import os
import json
import base64
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from src.ai_message_builder import (
    build_analysis_text_prompt,
    build_user_message_content,
)
from src.infrastructure.config.settings import AISettings
from src.infrastructure.config.env_manager import env_manager
from src.services.ai_request_compat import (
    CHAT_COMPLETIONS_API_MODE,
    RESPONSES_API_MODE,
    add_thinking_extra_body,
    build_ai_request_params,
    create_ai_response_async,
    is_chat_completions_api_unsupported_error,
    is_json_output_unsupported_error,
    is_responses_api_unsupported_error,
    is_temperature_unsupported_error,
    is_thinking_unsupported_error,
    remove_temperature_param,
)
from src.services.ai_response_parser import (
    EmptyAIResponseError,
    ensure_response_is_complete,
    extract_ai_response_content,
    parse_ai_response_json,
)
from src.services.ai_retry_policy import (
    AIRetryPolicy,
    format_retry_note,
    is_output_budget_error,
    is_transient_error,
)


def _normalize_no_proxy_host(part: str) -> str:
    """Return an httpx-safe NO_PROXY host, or the original entry if unchanged."""
    candidate = part.strip()
    if candidate.startswith("[") and "]" in candidate:
        inner, _, suffix = candidate[1:].partition("]")
        unwrapped = f"{inner}{suffix}"
        host = unwrapped.split("/", 1)[0]
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return part
        candidate = unwrapped

    if "/" not in candidate:
        return candidate

    host, _, _prefix = candidate.partition("/")
    try:
        ipaddress.IPv6Address(host)
    except ValueError:
        return candidate
    return host


def _sanitize_no_proxy_env() -> None:
    """Normalize IPv6 entries in NO_PROXY / no_proxy for httpx <= 0.28.1.

    Two forms are rejected while building proxy mounts:

    - ``::1/128`` is wrapped as ``[::1/128]``, and the prefix is parsed as a port.
    - ``[::1]`` is not recognized as IPv6, so it becomes ``all://*[::1]`` and
      fails with ``Invalid port: ':1]'``.

    httpx only does exact-host comparison, so stripping the prefix length and
    unwrapping bracketed IPv6 literals is safe. IPv4 CIDR entries are kept.

    See https://github.com/encode/httpx/pull/3741
    """
    for key in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(key)
        if not value:
            continue
        parts = [h.strip() for h in value.split(",")]
        cleaned = [_normalize_no_proxy_host(part) for part in parts]
        if cleaned != parts:
            os.environ[key] = ",".join(cleaned)


class AIClient:
    """AI 客户端封装"""

    def __init__(self):
        self.settings: Optional[AISettings] = None
        self.client: Optional[AsyncOpenAI] = None
        self.refresh()

    def _load_settings(self) -> None:
        load_dotenv(dotenv_path=env_manager.env_file, override=True)
        self.settings = AISettings()

    def refresh(self) -> None:
        self._load_settings()
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[AsyncOpenAI]:
        """初始化 OpenAI 客户端"""
        if not self.settings or not self.settings.is_configured():
            print("警告：AI 配置不完整，AI 功能将不可用")
            return None

        try:
            if self.settings.proxy_url:
                print(f"正在为 AI 请求使用代理: {self.settings.proxy_url}")
                os.environ['HTTP_PROXY'] = self.settings.proxy_url
                os.environ['HTTPS_PROXY'] = self.settings.proxy_url

            _sanitize_no_proxy_env()

            return AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(
                    self.settings.request_timeout_seconds,
                    connect=self.settings.connect_timeout_seconds,
                ),
                # SDK 内部重试会让日志里的尝试次数失真，重试统一由应用层负责。
                max_retries=self.settings.sdk_max_retries,
            )
        except Exception as e:
            print(f"初始化 AI 客户端失败: {e}")
            return None

    def is_available(self) -> bool:
        """检查 AI 客户端是否可用"""
        return self.client is not None

    async def close(self) -> None:
        """关闭底层异步客户端，避免事件循环结束后再触发清理。"""
        client = self.client
        self.client = None
        if client is None:
            return

        close = getattr(client, "close", None)
        if close is None:
            return
        await close()

    @staticmethod
    def encode_image(image_path: str) -> Optional[str]:
        """将图片编码为 Base64"""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"编码图片失败: {e}")
            return None

    async def analyze(
        self,
        product_data: Dict,
        image_paths: List[str],
        prompt_text: str
    ) -> Optional[Dict]:
        """
        分析商品数据

        Args:
            product_data: 商品数据
            image_paths: 图片路径列表
            prompt_text: 分析提示词

        Returns:
            分析结果
        """
        if not self.is_available():
            print("AI 客户端不可用")
            return None

        try:
            messages = self._build_messages(product_data, image_paths, prompt_text)
            item_info = product_data.get("商品信息", {}) or {}
            log_label = f"[AI分析] 商品 #{item_info.get('商品ID', 'N/A')}"
            response = await self._call_ai(messages, log_label=log_label)
            return self._parse_response(response)
        except Exception as e:
            print(f"AI 分析失败: {e}")
            return None

    def _build_messages(self, product_data: Dict, image_paths: List[str], prompt_text: str) -> List[Dict]:
        """构建 AI 消息"""
        product_json = json.dumps(product_data, ensure_ascii=False, indent=2)
        image_data_urls: List[str] = []
        for path in image_paths:
            base64_img = self.encode_image(path)
            if base64_img:
                image_data_urls.append(f"data:image/jpeg;base64,{base64_img}")

        text_prompt = build_analysis_text_prompt(
            product_json,
            prompt_text,
            include_images=bool(image_data_urls),
        )
        user_content = build_user_message_content(text_prompt, image_data_urls)
        return [{"role": "user", "content": user_content}]

    async def _call_ai(
        self,
        messages: List[Dict],
        *,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        enable_json_output: Optional[bool] = None,
        log_label: str = "",
    ) -> str:
        """调用 AI API（应用层统一负责尝试计数、退避与输出预算递增）"""
        settings = self.settings
        policy = AIRetryPolicy.from_settings(settings)
        api_mode = CHAT_COMPLETIONS_API_MODE
        use_response_format = (
            settings.enable_response_format
            if enable_json_output is None
            else enable_json_output
        )
        use_temperature = True
        use_thinking_param = settings.enable_thinking
        output_tokens = int(max_output_tokens or policy.base_output_tokens)
        prefix = f"{log_label} " if log_label else ""

        for attempt in policy.attempts():
            request_params = build_ai_request_params(
                api_mode,
                model=settings.model_name,
                messages=messages,
                temperature=temperature,
                max_output_tokens=output_tokens,
                enable_json_output=use_response_format,
            )
            if not use_temperature:
                request_params = remove_temperature_param(request_params)
            if use_thinking_param:
                request_params = add_thinking_extra_body(
                    request_params,
                    settings.thinking_enabled,
                    send_param=True,
                )

            try:
                response = await create_ai_response_async(
                    self.client,
                    api_mode,
                    request_params,
                )
                ensure_response_is_complete(response)
                return extract_ai_response_content(response)
            except Exception as exc:
                retryable = isinstance(exc, EmptyAIResponseError)
                if (
                    api_mode == CHAT_COMPLETIONS_API_MODE
                    and is_chat_completions_api_unsupported_error(exc)
                ):
                    api_mode = RESPONSES_API_MODE
                    retryable = True
                    print("当前服务未实现 Chat Completions API，正在自动回退到 Responses API")
                elif (
                    api_mode == RESPONSES_API_MODE
                    and is_responses_api_unsupported_error(exc)
                ):
                    api_mode = CHAT_COMPLETIONS_API_MODE
                    retryable = True
                    print("当前服务未实现 Responses API，正在自动回退到 Chat Completions API")
                if use_response_format and is_json_output_unsupported_error(exc):
                    use_response_format = False
                    retryable = True
                    print("当前模型不支持结构化 JSON 输出，正在自动重试并移除该参数")
                if use_temperature and is_temperature_unsupported_error(exc):
                    use_temperature = False
                    retryable = True
                    print("当前模型不支持 temperature 参数，正在自动重试并移除该参数")
                if use_thinking_param and is_thinking_unsupported_error(exc):
                    use_thinking_param = False
                    retryable = True
                    print("当前模型不支持 enable_thinking 参数，正在自动回退并移除该参数")
                if is_output_budget_error(exc):
                    retryable = True
                    escalated_tokens = policy.escalate_output_tokens(output_tokens)
                    if escalated_tokens > output_tokens:
                        output_tokens = escalated_tokens
                        print(f"{prefix}输出预算不足，下一次尝试将提高到 {output_tokens} tokens")
                if is_transient_error(exc):
                    retryable = True

                print(f"{prefix}{policy.describe_attempt(attempt)}失败: {exc}")
                if not retryable or not policy.has_next_attempt(attempt):
                    raise

                delay = policy.delay_seconds(attempt, exc)
                print(f"{prefix}{format_retry_note(policy, attempt, delay)}")
                if delay > 0:
                    await asyncio.sleep(delay)

        raise RuntimeError("AI 调用在兼容性重试后仍未返回结果")

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """解析 AI 响应"""
        try:
            return parse_ai_response_json(response_text)
        except json.JSONDecodeError:
            print(f"无法解析 AI 响应: {response_text[:100]}")
            return None
