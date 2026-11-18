import pytest
from src.services.ai_request_compat import (
    CHAT_COMPLETIONS_API_MODE,
    RESPONSES_API_MODE,
    add_reasoning_control,
    is_json_output_unsupported_error,
    is_responses_api_unsupported_error,
    is_temperature_unsupported_error,
    is_thinking_unsupported_error,
    normalize_thinking_param_style,
    remove_temperature_param,
    remove_thinking_param,
)


def test_is_temperature_unsupported_error_detects_unsupported_message():
    err = Exception("temperature is not supported by this gateway")
    assert is_temperature_unsupported_error(err) is True


def test_remove_temperature_param_removes_only_temperature():
    params = {"model": "x", "temperature": 0.5, "max_output_tokens": 128}
    result = remove_temperature_param(params)

    assert "temperature" not in result
    assert result["model"] == "x"
    assert result["max_output_tokens"] == 128


def test_is_responses_api_unsupported_error_detects_gemini_plain_404():
    class _Resp:
        text = ""

    class _Err(Exception):
        status_code = 404
        body = ""
        response = _Resp()

        def __str__(self):
            return "Error code: 404"

    assert is_responses_api_unsupported_error(_Err()) is True


# -- is_json_output_unsupported_error tests --


def test_json_output_error_detected_via_body_param_response_format():
    """Vercel AI Gateway returns 400 with param='response_format'."""

    class _Err(Exception):
        body = {
            "message": "Invalid input",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "invalid_request_error",
        }

    assert is_json_output_unsupported_error(_Err()) is True


def test_json_output_error_detected_via_body_param_response_format_type():
    class _Err(Exception):
        body = {
            "message": "Invalid input",
            "param": "response_format.type",
        }

    assert is_json_output_unsupported_error(_Err()) is True


def test_json_output_error_detected_via_legacy_string_matching():
    err = Exception(
        "response_format.type is not supported by this model"
    )
    assert is_json_output_unsupported_error(err) is True


def test_json_output_error_not_triggered_by_unrelated_400():
    class _Err(Exception):
        body = {
            "message": "Invalid input",
            "param": "messages",
        }

    assert is_json_output_unsupported_error(_Err()) is False


def test_json_output_error_not_triggered_without_body():
    err = Exception("some random 400 error")
    assert is_json_output_unsupported_error(err) is False


def test_add_reasoning_control_writes_extra_body_style_by_default():
    params = add_reasoning_control(
        {"model": "x", "messages": []},
        CHAT_COMPLETIONS_API_MODE,
        enabled=False,
    )

    assert params["extra_body"] == {"enable_thinking": False}


def test_add_reasoning_control_writes_chat_template_kwargs_style():
    params = add_reasoning_control(
        {"model": "x", "messages": []},
        CHAT_COMPLETIONS_API_MODE,
        style="chat_template_kwargs",
        enabled=False,
    )

    assert params["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_add_reasoning_control_keeps_existing_chat_template_kwargs():
    params = add_reasoning_control(
        {"extra_body": {"chat_template_kwargs": {"add_generation_prompt": True}}},
        CHAT_COMPLETIONS_API_MODE,
        style="chat_template_kwargs",
        enabled=False,
    )

    assert params["extra_body"]["chat_template_kwargs"] == {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_add_reasoning_control_adds_reasoning_effort_to_responses_api():
    params = add_reasoning_control(
        {"model": "x", "input": []},
        RESPONSES_API_MODE,
        effort="low",
    )

    assert params["reasoning"] == {"effort": "low"}
    assert "extra_body" not in params


def test_add_reasoning_control_combines_effort_with_style_for_chat_completions():
    params = add_reasoning_control(
        {"model": "x", "messages": []},
        CHAT_COMPLETIONS_API_MODE,
        style="chat_template_kwargs",
        enabled=False,
        effort="minimal",
    )

    assert params["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "minimal",
    }


def test_add_reasoning_control_skips_params_when_nothing_is_configured():
    params = add_reasoning_control({"model": "x", "messages": []}, CHAT_COMPLETIONS_API_MODE)

    assert "extra_body" not in params


def test_normalize_thinking_param_style_rejects_unknown_style():
    with pytest.raises(ValueError):
        normalize_thinking_param_style("unknown_style")


def test_is_thinking_unsupported_error_detects_chat_template_kwargs_rejection():
    err = Exception("chat_template_kwargs is not supported by this gateway")
    assert is_thinking_unsupported_error(err) is True


def test_is_thinking_unsupported_error_detects_body_param_reasoning_effort():
    class _Err(Exception):
        body = {"param": "reasoning_effort"}

    assert is_thinking_unsupported_error(_Err()) is True


def test_remove_thinking_param_removes_all_reasoning_control_keys():
    params = {
        "model": "x",
        "messages": [],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "low",
        },
        "reasoning": {"effort": "low"},
    }

    result = remove_thinking_param(params)

    assert "extra_body" not in result
    assert "reasoning" not in result
    assert result["model"] == "x"
