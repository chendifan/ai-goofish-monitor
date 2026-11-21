import asyncio
from pathlib import Path
from types import SimpleNamespace

import requests

import src.ai_handler as ai_handler
from src.services.image_type_detector import detect_image_mime


def test_download_all_images_runs_with_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_handler, "IMAGE_SAVE_DIR", str(tmp_path / "images"))

    active_downloads = 0
    max_active_downloads = 0

    async def fake_download(url, save_path):
        nonlocal active_downloads, max_active_downloads
        active_downloads += 1
        max_active_downloads = max(max_active_downloads, active_downloads)
        await asyncio.sleep(0.02)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text("ok", encoding="utf-8")
        active_downloads -= 1
        return save_path

    monkeypatch.setattr(ai_handler, "_download_single_image", fake_download)

    async def run():
        return await ai_handler.download_all_images(
            "product-1",
            [
                "https://example.com/1.jpg",
                "https://example.com/2.jpg",
                "https://example.com/3.jpg",
            ],
            task_name="demo",
            concurrency=3,
        )

    paths = asyncio.run(run())
    assert len(paths) == 3
    assert max_active_downloads == 3


class FakeResponse:
    def __init__(self, content_type, body, status_code=200):
        self.headers = {"Content-Type": content_type}
        self._body = body
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error", response=self)

    def iter_content(self, chunk_size=8192):
        yield self._body

    def close(self):
        self.closed = True


def _install_fake_response(monkeypatch, content_type, body):
    response = FakeResponse(content_type, body)
    monkeypatch.setattr(ai_handler, "requests", SimpleNamespace(get=lambda url, **kwargs: response))
    return response


def test_download_single_image_skips_cdn_placeholder_image(tmp_path, monkeypatch, capsys):
    jpeg_start = bytes([0xff, 0xd8])
    placeholder = b"GIF89a" + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + bytes(40)
    response = _install_fake_response(monkeypatch, "image/gif", placeholder)
    save_path = tmp_path / "product_dead_1_x.jpg"

    result = asyncio.run(ai_handler._download_single_image("http://img.example/1.jpg", str(save_path)))
    assert result is None
    assert not save_path.exists()
    assert "占位图" in capsys.readouterr().out
    assert response.closed is True
    assert detect_image_mime(str(tmp_path / "missing.png")) == "image/jpeg"


def test_download_single_image_skips_non_image_response(tmp_path, monkeypatch):
    _install_fake_response(monkeypatch, "text/html", b"<html>" * 300)
    save_path = tmp_path / "product_html.jpg"

    assert asyncio.run(ai_handler._download_single_image("http://img.example/2.jpg", str(save_path))) is None
    assert not save_path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_single_image_writes_image_atomically(tmp_path, monkeypatch):
    jpeg_start = bytes([0xff, 0xd8])
    _install_fake_response(monkeypatch, "image/jpeg", jpeg_start + b"x" * 1200)
    save_path = tmp_path / "product_real.jpg"

    saved = asyncio.run(ai_handler._download_single_image("http://img.example/3.jpg", str(save_path)))
    assert saved == str(save_path)
    assert save_path.read_bytes().startswith(jpeg_start)
    assert not list(tmp_path.glob("*.part"))
    assert detect_image_mime(str(save_path)) == "image/jpeg"
