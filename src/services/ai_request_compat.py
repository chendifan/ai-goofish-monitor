"""AI 请求兼容性辅助逻辑。"""

import copy
from typing import Any, Dict, Iterable, List, Optional


RESPONSES_API_MODE = "responses"
CHAT_COMPLETIONS_API_MODE = "chat_completions"
INPUT_TEXT_TYPE = "input_text"
INPUT_IMAGE_TYPE = "input_image"
IMAGE_DETAIL_AUTO = "auto"
JSON_OUTPUT_TYPE = "json_object"
UNSUPPORTED_JSON_OUTPUT_MARKERS = (
    "not supported by this model",
    "json_object",
    "json_schema",
    "text.format",
    "response_format.type",
)
RESPONSES_API_UNSUPPORTED_MARKERS = (
    "404 page not found",
    "page not found",
    "/responses",
    "/v1/responses",
)
CHAT_COMPLETIONS_API_UNSUPPORTED_MARKERS = (
    "404 page not found",
    "page not found",
    "/chat/completions",
    "/v1/chat/completions",
)
UNSUPPORTED_TEMPERATURE_MARKERS = (
    "temperature",
    "sampling temperature",
)
UNSUPPORTED_THINKING_MARKERS = (
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_effort",
    "thinking",
    "reasoning",
)
REASONING_CONTROL_PARAM_NAMES = (
    "enable_thinking",
    "extra_body.enable_thinking",
    "chat_template_kwargs",
    "extra_body.chat_template_kwargs",
    "reasoning_effort",
    "extra_body.reasoning_effort",
    "reasoning",
)
THINKING_PARAM_STYLE_EXTRA_BODY = "extra_body"
THINKING_PARAM_STYLE_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
THINKING_PARAM_STYLES = (
    THINKING_PARAM_STYLE_EXTRA_BODY,
    THINKING_PARAM_STYLE_CHAT_TEMPLATE_KWARGS,
)


def normalize_thinking_param_style(value: Any) -> str:
    """校验思考开关的参数形态，未知形态直接报错而不是静默忽略。"""
    style = str(value or THINKING_PARAM_STYLE_EXTRA_BODY).strip().lower()
    if style not in THINKING_PARAM_STYLES:
        raise ValueError(
            "不支持的思考参数形态: "
            f"{value}（可选: {', '.join(THINKING_PARAM_STYLES)}）"
        )
    return style


def add_reasoning_control(
    request_params: Dict[str, Any],
    api_mode: str = CHAT_COMPLETIONS_API_MODE,
    *,
    style: str = THINKING_PARAM_STYLE_EXTRA_BODY,
    enabled: Optional[bool] = None,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """按需附加思考开关与推理强度参数。

    不同网关接受的形态不同：vLLM 类服务通常只认 chat_template_kwargs，
    多数 OpenAI 兼容中转则只认 extra_body 顶层或 Responses API 的 reasoning。
    """
    next_params = dict(request_params)
    style = normalize_thinking_param_style(style)

    if api_mode == RESPONSES_API_MODE:
        if effort:
            next_params["reasoning"] = {"effort": str(effort)}
        return next_params

    if enabled is None and not effort:
        return next_params

    extra_body = dict(next_params.get("extra_body") or {})
    if enabled is not None:
        if style == THINKING_PARAM_STYLE_CHAT_TEMPLATE_KWARGS:
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = bool(enabled)
            extra_body["chat_template_kwargs"] = template_kwargs
        else:
            extra_body["enable_thinking"] = bool(enabled)
    if effort:
        extra_body["reasoning_effort"] = str(effort)

    if extra_body:
        next_params["extra_body"] = extra_body
    return next_params


def build_responses_input(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 Chat Completions 风格的消息转换为 Responses API 输入。"""
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        input_items.append(
            {
                "role": role,
                "content": _build_input_content(message.get("content")),
            }
        )
    return input_items


def add_json_text_format(
    request_params: Dict[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    """按需附加 Responses API 的结构化 JSON 输出参数。"""
    next_params = dict(request_params)
    if not enabled:
        return next_params

    text_config = dict(next_params.get("text") or {})
    text_config["format"] = {"type": JSON_OUTPUT_TYPE}
    next_params["text"] = text_config
    return next_params


def add_json_response_format(
    request_params: Dict[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    """按需附加 Chat Completions 的 JSON 输出参数。"""
    next_params = dict(request_params)
    if enabled:
        next_params["response_format"] = {"type": JSON_OUTPUT_TYPE}
    return next_params


def is_json_output_unsupported_error(error: Exception) -> bool:
    """识别模型或网关不支持结构化 JSON 输出参数的错误。"""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("param") in (
        "response_format",
        "response_format.type",
    ):
        return True

    message = str(error)
    return (
        "not supported" in message.lower()
        and any(marker in message for marker in UNSUPPORTED_JSON_OUTPUT_MARKERS)
    )


def is_responses_api_unsupported_error(error: Exception) -> bool:
    """识别 OpenAI 兼容服务未实现 Responses API 的错误。"""
    return _is_api_unsupported_error(error, RESPONSES_API_UNSUPPORTED_MARKERS)


def is_chat_completions_api_unsupported_error(error: Exception) -> bool:
    """识别 OpenAI 兼容服务未实现 Chat Completions API 的错误。"""
    return _is_api_unsupported_error(error, CHAT_COMPLETIONS_API_UNSUPPORTED_MARKERS)


def build_ai_request_params(
    api_mode: str,
    *,
    model: str,
    messages: Iterable[Dict[str, Any]],
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    enable_json_output: bool = False,
) -> Dict[str, Any]:
    """根据 API 模式构建请求参数。"""
    request_params = {"model": model}
    if api_mode == RESPONSES_API_MODE:
        request_params["input"] = build_responses_input(messages)
        if max_output_tokens is not None:
            request_params["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            request_params["temperature"] = temperature
        return add_json_text_format(request_params, enable_json_output)

    if api_mode == CHAT_COMPLETIONS_API_MODE:
        request_params["messages"] = copy.deepcopy(list(messages))
        if max_output_tokens is not None:
            request_params["max_tokens"] = max_output_tokens
        if temperature is not None:
            request_params["temperature"] = temperature
        return add_json_response_format(request_params, enable_json_output)

    raise ValueError(f"不支持的 AI API 模式: {api_mode}")


async def create_ai_response_async(
    client: Any,
    api_mode: str,
    request_params: Dict[str, Any],
) -> Any:
    """根据 API 模式发起异步请求。"""
    if api_mode == RESPONSES_API_MODE:
        return await client.responses.create(**request_params)
    if api_mode == CHAT_COMPLETIONS_API_MODE:
        return await client.chat.completions.create(**request_params)
    raise ValueError(f"不支持的 AI API 模式: {api_mode}")


def create_ai_response_sync(
    client: Any,
    api_mode: str,
    request_params: Dict[str, Any],
) -> Any:
    """根据 API 模式发起同步请求。"""
    if api_mode == RESPONSES_API_MODE:
        return client.responses.create(**request_params)
    if api_mode == CHAT_COMPLETIONS_API_MODE:
        return client.chat.completions.create(**request_params)
    raise ValueError(f"不支持的 AI API 模式: {api_mode}")


def is_temperature_unsupported_error(error: Exception) -> bool:
    """识别模型或中转站不支持 temperature 参数的错误。"""
    message = str(error).lower()
    return (
        "not supported" in message
        or "unsupported" in message
        or "invalid" in message
        or "参数错误" in message
    ) and any(marker in message for marker in UNSUPPORTED_TEMPERATURE_MARKERS)


def remove_temperature_param(request_params: Dict[str, Any]) -> Dict[str, Any]:
    """移除 temperature 参数，适配不支持采样温度的模型网关。"""
    next_params = dict(request_params)
    next_params.pop("temperature", None)
    return next_params


def is_thinking_unsupported_error(error: Exception) -> bool:
    """识别模型或网关不支持思考开关/推理强度参数的错误。"""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("param") in REASONING_CONTROL_PARAM_NAMES:
        return True

    message = str(error).lower()
    return (
        "not supported" in message
        or "unsupported" in message
        or "invalid" in message
        or "参数错误" in message
    ) and any(marker in message for marker in UNSUPPORTED_THINKING_MARKERS)


def remove_thinking_param(request_params: Dict[str, Any]) -> Dict[str, Any]:
    """移除思考开关与推理强度参数，适配不支持这些参数的模型网关。"""
    next_params = dict(request_params)
    next_params.pop("reasoning", None)
    extra_body = next_params.get("extra_body")
    if isinstance(extra_body, dict):
        cleaned = dict(extra_body)
        cleaned.pop("enable_thinking", None)
        cleaned.pop("chat_template_kwargs", None)
        cleaned.pop("reasoning_effort", None)
        if cleaned:
            next_params["extra_body"] = cleaned
        else:
            del next_params["extra_body"]
    return next_params


def _is_api_unsupported_error(
    error: Exception,
    markers: tuple[str, ...],
) -> bool:
    message = str(error).lower()
    if any(marker in message for marker in markers):
        return True

    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    response = getattr(error, "response", None)
    response_text = getattr(response, "text", None) if response else None
    return (
        status_code == 404
        and message.strip() == "error code: 404"
        and not body
        and not response_text
    )


def _build_input_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": INPUT_TEXT_TYPE, "text": content}]
    if not isinstance(content, list):
        raise ValueError(f"AI消息内容类型不受支持: {type(content).__name__}")

    return [_coerce_content_item(item) for item in content]


def _coerce_content_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"AI消息片段类型不受支持: {type(item).__name__}")

    item_type = item.get("type")
    if item_type in {"text", INPUT_TEXT_TYPE}:
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("文本消息片段缺少 text 字段。")
        return {"type": INPUT_TEXT_TYPE, "text": text}

    if item_type in {"image_url", INPUT_IMAGE_TYPE}:
        return _build_image_input_item(item)

    raise ValueError(f"不支持的 AI 消息片段类型: {item_type}")


def _build_image_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_image = item.get("image_url")
    if isinstance(raw_image, dict):
        image_url = raw_image.get("url")
    else:
        image_url = raw_image

    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("图片消息片段缺少有效的 image_url。")

    return {
        "type": INPUT_IMAGE_TYPE,
        "image_url": image_url,
        "detail": item.get("detail", IMAGE_DETAIL_AUTO),
    }
