import asyncio
from types import SimpleNamespace

import requests

from src.services.result_storage_service import load_all_result_records
from src.utils import (
    describe_error_body,
    error_status_code,
    format_registration_days,
    get_link_unique_key,
    is_permanent_client_error,
    retry_on_failure,
    safe_get,
    save_to_jsonl,
)


def test_safe_get_nested_and_default():
    data = {"a": {"b": [{"c": "value"}]}}
    assert asyncio.run(safe_get(data, "a", "b", 0, "c")) == "value"
    assert asyncio.run(safe_get(data, "a", "b", 1, "c", default="missing")) == "missing"


def test_format_registration_days():
    assert format_registration_days(400).startswith("\u6765\u95f2\u9c7c")
    assert format_registration_days(-1) == "\u672a\u77e5"


def test_get_link_unique_key():
    link = "https://www.goofish.com/item?id=123&foo=bar"
    assert get_link_unique_key(link) == "https://www.goofish.com/item?id=123"


def test_save_to_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = {
        "爬取时间": "2026-01-01T10:00:00",
        "搜索关键字": "sony a7m4",
        "任务名称": "Sony A7M4",
        "商品信息": {
            "商品ID": "1",
            "商品标题": "Sony A7M4",
            "商品链接": "https://www.goofish.com/item?id=1",
            "当前售价": "¥10000",
        },
    }

    ok = asyncio.run(save_to_jsonl(record, keyword="sony a7m4"))
    assert ok is True

    records = asyncio.run(
        load_all_result_records(
            "sony_a7m4_full_data.jsonl",
            ai_recommended_only=False,
            keyword_recommended_only=False,
            sort_by="crawl_time",
            sort_order="asc",
        )
    )
    assert records == [record]


def test_error_status_code_falls_back_to_response_for_requests_http_error():
    response = SimpleNamespace(status_code=404, headers={}, content=b"")
    error = requests.exceptions.HTTPError("404 Client Error", response=response)
    assert error_status_code(error) == 404
    assert is_permanent_client_error(error) is True


def test_permanent_client_error_keeps_retryable_status_codes():
    gateways = [429, 502, 504]
    for code in gateways:
        error = requests.exceptions.HTTPError(
            f"{code} error",
            response=SimpleNamespace(status_code=code, headers={}, content=b""),
        )
        assert is_permanent_client_error(error) is False


def test_describe_error_body_skips_binary_responses():
    response = SimpleNamespace(
        status_code=404,
        headers={"Content-Type": "image/gif"},
        content=b"GIF89a\x01\x00\x01\x00",
        text="binary decoded as text",
    )
    note = describe_error_body(response)
    assert "image/gif" in note
    assert "binary decoded as text" not in note


def test_retry_on_failure_stops_on_permanent_client_error(capsys):
    calls = []
    response = SimpleNamespace(
        status_code=404,
        headers={"Content-Type": "image/gif"},
        content=b"GIF89a placeholder",
        text="GIF89a placeholder",
    )
    error = requests.exceptions.HTTPError("404 Client Error", response=response)

    @retry_on_failure(retries=3, delay=0)
    async def download():
        calls.append(1)
        raise error

    assert asyncio.run(download()) is None
    assert len(calls) == 1
    captured = capsys.readouterr().out
    assert "Status Code): 404" in captured
    assert "GIF89a placeholder" not in captured


def test_retry_on_failure_still_retries_server_errors():
    calls = []

    @retry_on_failure(retries=2, delay=0)
    async def download():
        calls.append(1)
        raise requests.exceptions.HTTPError(
            "502 error",
            response=SimpleNamespace(
                status_code=502,
                headers={"Content-Type": "text/plain"},
                content=b"bad gateway",
                text="bad gateway",
            ),
        )

    assert asyncio.run(download()) is None
    assert len(calls) == 2


def test_describe_error_body_uses_content_length_when_body_is_closed():
    response = SimpleNamespace(
        status_code=404,
        headers={"Content-Type": "image/gif", "Content-Length": "49"},
        content=b"",
        text="",
    )
    assert "49 字节" in describe_error_body(response)
