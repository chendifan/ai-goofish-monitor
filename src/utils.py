import asyncio
import json
import math
import os
import random
import re
import glob
from datetime import datetime
from functools import wraps
from urllib.parse import quote
from typing import Optional

from openai import APIStatusError
from requests.exceptions import HTTPError

from src.services.result_storage_service import save_result_record


def error_status_code(error) -> Optional[int]:
    """requests 的 HTTPError 没有 status_code 属性，状态码要从 response 上取。"""
    code = getattr(error, "status_code", None)
    if code is None:
        code = getattr(getattr(error, "response", None), "status_code", None)
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def is_permanent_client_error(error) -> bool:
    """4xx 里除 408 和 429 之外重试没有意义。"""
    code = error_status_code(error)
    return code is not None and 400 <= code < 500 and code not in {408, 429}


def describe_error_body(response) -> str:
    """文本类响应才写进日志，图片之类的二进制内容只记录类型和大小。"""
    if response is None:
        return ""
    headers = getattr(response, "headers", None) or {}
    content_type = str(headers.get("Content-Type", "")).split(";")[0]
    body = getattr(response, "content", b"") or b""
    if content_type.startswith("text/") or content_type in {"application/json", "application/problem+json"}:
        text = response.text[:300]
        suffix = "..." if len(response.text) > 300 else ""
        return f"响应内容 ({content_type}): {text}{suffix}"
    size = len(body) or int(headers.get("Content-Length") or 0)
    return f"响应类型 {content_type or '未知'}，二进制内容 {size} 字节未写入日志"


def retry_on_failure(retries=3, delay=5):
    """
    一个通用的异步重试装饰器，增加了对HTTP错误的详细日志记录。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (APIStatusError, HTTPError) as e:
                    print(f"函数 {func.__name__} 第 {i + 1}/{retries} 次尝试失败，发生HTTP错误。")
                    status_code = error_status_code(e)
                    if status_code is not None:
                        print(f"  - 状态码 (Status Code): {status_code}")
                    body_note = describe_error_body(getattr(e, "response", None))
                    if body_note:
                        print(f"  - {body_note}")
                    if is_permanent_client_error(e):
                        print("  - 客户端错误，重试没有意义，直接放弃。")
                        return None
                except json.JSONDecodeError as e:
                    print(f"函数 {func.__name__} 第 {i + 1}/{retries} 次尝试失败: JSON解析错误 - {e}")
                except Exception as e:
                    print(f"函数 {func.__name__} 第 {i + 1}/{retries} 次尝试失败: {type(e).__name__} - {e}")

                if i < retries - 1:
                    print(f"将在 {delay} 秒后重试...")
                    await asyncio.sleep(delay)

            print(f"函数 {func.__name__} 在 {retries} 次尝试后彻底失败。")
            return None
        return wrapper
    return decorator


async def safe_get(data, *keys, default="暂无"):
    """安全获取嵌套字典值"""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


async def random_sleep(min_seconds: float, max_seconds: float):
    """异步等待一个在指定范围内的随机时间。"""
    delay = random.uniform(min_seconds, max_seconds)
    print(f"   [延迟] 等待 {delay:.2f} 秒... (范围: {min_seconds}-{max_seconds}s)")
    await asyncio.sleep(delay)


def log_time(message: str, prefix: str = "") -> None:
    """在日志前加上 YY-MM-DD HH:MM:SS 时间戳的简单打印。"""
    try:
        ts = datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
    except Exception:
        ts = "--:--:--"
    print(f"[{ts}] {prefix}{message}")


def sanitize_filename(value: str) -> str:
    """生成安全的文件名片段。"""
    if not value:
        return "task"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "task"


def build_task_log_path(task_id: int, task_name: str) -> str:
    """生成任务日志路径（包含任务名）。"""
    safe_name = sanitize_filename(task_name)
    filename = f"{safe_name}_{task_id}.log"
    return os.path.join("logs", filename)


def resolve_task_log_path(task_id: int, task_name: str) -> str:
    """优先使用任务名生成日志路径，不存在时回退为按 ID 匹配。"""
    primary_path = build_task_log_path(task_id, task_name)
    if os.path.exists(primary_path):
        return primary_path
    pattern = os.path.join("logs", f"*_{task_id}.log")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return primary_path


def convert_goofish_link(url: str) -> str:
    """
    将Goofish商品链接转换为只包含商品ID的手机端格式。
    """
    match_first_link = re.search(r'item\?id=(\d+)', url)
    if match_first_link:
        item_id = match_first_link.group(1)
        bfp_json = f'{{"id":{item_id}}}'
        return f"https://pages.goofish.com/sharexy?loadingVisible=false&bft=item&bfs=idlepc.item&spm=a21ybx.item.0.0&bfp={quote(bfp_json)}"
    return url


def get_link_unique_key(link: str) -> str:
    """截取链接中第一个"&"之前的内容作为唯一标识依据。"""
    return link.split('&', 1)[0]


async def save_to_jsonl(data_record: dict, keyword: str):
    """兼容旧调用名，实际将结果写入 SQLite。"""
    try:
        return await save_result_record(data_record, keyword)
    except Exception as e:
        print(f"写入 SQLite 结果记录出错: {e}")
        return False


def format_registration_days(total_days: int) -> str:
    """
    将总天数格式化为“X年Y个月”的字符串。
    """
    if not isinstance(total_days, int) or total_days <= 0:
        return '未知'

    DAYS_IN_YEAR = 365.25
    DAYS_IN_MONTH = DAYS_IN_YEAR / 12

    years = math.floor(total_days / DAYS_IN_YEAR)
    remaining_days = total_days - (years * DAYS_IN_YEAR)
    months = round(remaining_days / DAYS_IN_MONTH)

    if months == 12:
        years += 1
        months = 0

    if years > 0 and months > 0:
        return f"来闲鱼{years}年{months}个月"
    elif years > 0 and months == 0:
        return f"来闲鱼{years}年整"
    elif years == 0 and months > 0:
        return f"来闲鱼{months}个月"
    else:
        return "来闲鱼不足一个月"
