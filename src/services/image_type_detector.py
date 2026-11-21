"""图片类型判定：用于跳过 CDN 占位图并按真实格式标注 data URL 的 MIME。"""

MIN_IMAGE_BYTES = 1024


def image_mime_from_header(head: bytes) -> str:
    """按文件头判断图片类型，无法判断时退回 image/jpeg。"""
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":
        return "image/heic"
    return "image/jpeg"


def detect_image_mime(path: str) -> str:
    """读取本地图片文件头，构造 data URL 时按真实类型标注 MIME。"""
    try:
        with open(path, "rb") as image_file:
            head = image_file.read(12)
    except OSError:
        return "image/jpeg"
    return image_mime_from_header(head)


def is_placeholder_image(head: bytes) -> bool:
    """识别 1x1 占位图，CDN 对已删除的图片会返回这种 GIF。"""
    if head[:6] not in (b"GIF87a", b"GIF89a") or len(head) < 10:
        return False
    width = int.from_bytes(head[6:8], "little")
    height = int.from_bytes(head[8:10], "little")
    return width <= 2 and height <= 2
