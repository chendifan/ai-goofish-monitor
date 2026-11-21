import asyncio
from types import SimpleNamespace

import pytest

import src.ai_handler as ai_handler
import src.config as app_config


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setenv("AI_RETRY_BASE_DELAY_SECONDS", "0")


def _build_fake_client(responses_create_impl, chat_create_impl=None):
    responses = SimpleNamespace(create=responses_create_impl)
    chat = SimpleNamespace(
        completions=SimpleNamespace(create=chat_create_impl or responses_create_impl)
    )
    return SimpleNamespace(responses=responses, chat=chat)


def test_get_ai_analysis_stops_after_internal_retries_when_content_is_none(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    call_count = {"value": 0}

    async def fake_create(**_kwargs):
        call_count["value"] += 1
        return SimpleNamespace(output_text="")

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    with pytest.raises(ValueError, match="AI响应内容为空"):
        asyncio.run(
            ai_handler.get_ai_analysis(
                {"商品信息": {"商品ID": "1", "商品标题": "测试商品"}},
                image_paths=[],
                prompt_text="请输出 JSON",
            )
        )

    assert call_count["value"] == 4


def test_get_ai_analysis_returns_parsed_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    call_count = {"value": 0}

    async def fake_create(**_kwargs):
        call_count["value"] += 1
        return SimpleNamespace(
            output_text=(
                '{"prompt_version":"v1","is_recommended":true,'
                '"reason":"ok","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}'
            )
        )

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "2", "商品标题": "测试商品2"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["is_recommended"] is True
    assert call_count["value"] == 1


def test_get_ai_analysis_retries_without_structured_output_when_model_rejects_it(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    request_history = []

    async def fake_create(**kwargs):
        request_history.append(kwargs)
        if len(request_history) == 1:
            raise Exception(
                "Error code: 400 - {'error': {'code': 'InvalidParameter', "
                "'message': 'The parameter `response_format.type` specified in "
                "the request are not valid: `json_object` is not supported by "
                "this model.', 'param': 'response_format.type'}}"
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"prompt_version":"v1","is_recommended":true,'
                            '"reason":"ok","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "3", "商品标题": "测试商品3"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["reason"] == "ok"
    assert request_history[0]["messages"][0]["role"] == "user"
    assert request_history[0]["response_format"]["type"] == "json_object"
    assert "response_format" not in request_history[1]
    assert ai_handler.ENABLE_RESPONSE_FORMAT is True


def test_get_ai_analysis_falls_back_to_responses_when_chat_completions_api_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    request_history = []

    async def fake_chat_create(**kwargs):
        request_history.append(("chat", kwargs))
        raise Exception("Error code: 404 - page not found")

    async def fake_responses_create(**kwargs):
        request_history.append(("responses", kwargs))
        if len([item for item in request_history if item[0] == "responses"]) == 1:
            raise Exception(
                "Error code: 400 - {'error': {'code': 'InvalidParameter', "
                "'message': 'The parameter `text.format.type` specified in "
                "the request are not valid: `json_object` is not supported by "
                "this model.', 'param': 'text.format.type'}}"
            )
        return SimpleNamespace(
            output_text=(
                '{"prompt_version":"v1","is_recommended":true,'
                '"reason":"ok","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}'
            )
        )

    monkeypatch.setattr(
        ai_handler,
        "client",
        _build_fake_client(fake_responses_create, fake_chat_create),
    )
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "4", "商品标题": "测试商品4"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["reason"] == "ok"
    assert request_history[0][0] == "chat"
    assert request_history[0][1]["messages"][0]["role"] == "user"
    assert request_history[1][0] == "responses"
    assert request_history[1][1]["text"]["format"]["type"] == "json_object"
    assert request_history[2][0] == "responses"
    assert "text" not in request_history[2][1]


def test_get_ai_analysis_retries_without_temperature_when_gateway_rejects_it(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    request_history = []

    async def fake_create(**kwargs):
        request_history.append(kwargs)
        if len(request_history) == 1:
            raise Exception("temperature is unsupported for this model")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"prompt_version":"v1","is_recommended":true,'
                            '"reason":"ok","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "4", "商品标题": "测试商品4"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["reason"] == "ok"
    assert request_history[0]["temperature"] == 0.1
    assert "temperature" not in request_history[1]


def test_get_ai_analysis_uses_first_json_object_when_model_returns_multiple_objects(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    async def fake_create(**_kwargs):
        return SimpleNamespace(
            output_text="""```json
{"prompt_version":"v1","is_recommended":true,"reason":"first","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}
{"prompt_version":"v1","is_recommended":false,"reason":"second","risk_tags":[],"criteria_analysis":{"seller_type":"商家"}}
```"""
        )

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")
    monkeypatch.setattr(ai_handler, "ENABLE_RESPONSE_FORMAT", True)
    monkeypatch.setattr(app_config, "ENABLE_RESPONSE_FORMAT", True)

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "5", "商品标题": "测试商品5"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["is_recommended"] is True
    assert result["reason"] == "first"


def test_get_ai_analysis_escalates_output_budget_after_truncation_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "4000")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS_CAP", "16000")
    request_history = []

    async def fake_create(**kwargs):
        request_history.append(kwargs)
        if len(request_history) <= 2:
            raise Exception(
                "Error code: 502 - {'error': {'type': 'structured_output_failed', 'code': "
                "'structured_output_failed', 'message': 'structured output was incomplete "
                "(finish_reason=length); increase the output budget'}}"
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"prompt_version":"v1","is_recommended":true,'
                            '"reason":"ok","risk_tags":[],"criteria_analysis":{"seller_type":"个人"}}'
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "9", "商品标题": "测试商品9"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["reason"] == "ok"
    assert [item["max_tokens"] for item in request_history] == [4000, 8000, 16000]

    captured = capsys.readouterr().out
    assert "商品 #9 第 1/4 次尝试失败" in captured
    assert "商品 #9 第 2/4 次尝试失败" in captured
    assert "输出预算不足，下一次尝试将提高到 8000 tokens" in captured
    assert "商品 #9 第 3/4 次尝试成功" in captured


def test_attempt_counts_are_per_product_when_analyses_run_concurrently(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_MAX_ATTEMPTS", "3")

    async def fake_create(**_kwargs):
        raise Exception("Request timed out.")

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")

    async def _run():
        return await asyncio.gather(
            ai_handler.get_ai_analysis(
                {"商品信息": {"商品ID": "10", "商品标题": "测试商品10"}},
                image_paths=[],
                prompt_text="请输出 JSON",
            ),
            ai_handler.get_ai_analysis(
                {"商品信息": {"商品ID": "11", "商品标题": "测试商品11"}},
                image_paths=[],
                prompt_text="请输出 JSON",
            ),
            return_exceptions=True,
        )

    results = asyncio.run(_run())

    assert all(isinstance(result, Exception) for result in results)
    captured = capsys.readouterr().out
    for product_id in ("10", "11"):
        for attempt in (1, 2, 3):
            assert f"商品 #{product_id} 第 {attempt}/3 次尝试失败" in captured


def test_get_ai_analysis_separates_cap_reached_from_bad_json(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "32000")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS_CAP", "32000")

    call_count = {"value": 0}
    truncation = Exception(
        "Error code: 502 - structured output was incomplete (finish_reason=length); "
        "increase the output budget"
    )
    bad_json = Exception(
        "Error code: 502 - model did not return valid JSON: Expecting ',' delimiter: "
        "line 1 column 2018"
    )
    good_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"prompt_version":"v1","is_recommended":true,'
                        '"reason":"ok","risk_tags":[],'
                        '"criteria_analysis":{"seller_type":"个人"}}'
                    )
                ),
                finish_reason="stop",
            )
        ]
    )

    async def fake_create(**_kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise truncation
        if call_count["value"] == 2:
            raise bad_json
        return good_response

    monkeypatch.setattr(ai_handler, "client", _build_fake_client(fake_create))
    monkeypatch.setattr(ai_handler, "MODEL_NAME", "fake-model")

    result = asyncio.run(
        ai_handler.get_ai_analysis(
            {"商品信息": {"商品ID": "9", "商品标题": "测试商品9"}},
            image_paths=[],
            prompt_text="请输出 JSON",
        )
    )

    assert result["reason"] == "ok"
    captured = capsys.readouterr().out
    assert "商品 #9 输出预算不足，已经在上限 32000 tokens" in captured
    assert "商品 #9 JSON 校验失败" in captured
    assert "商品 #9 第 3/4 次尝试成功" in captured
