import asyncio
import base64
import json
import os
import re
import sys
import shutil
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import requests

# 设置标准输出编码为UTF-8，解决Windows控制台编码问题
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from src.config import (
    AI_DEBUG_MODE,
    IMAGE_DOWNLOAD_HEADERS,
    IMAGE_SAVE_DIR,
    TASK_IMAGE_DIR_PREFIX,
    MODEL_NAME,
    ENABLE_RESPONSE_FORMAT,
    ENABLE_THINKING,
    AI_THINKING_PARAM_STYLE,
    AI_REASONING_EFFORT,
    client,
)
from src.ai_message_builder import (
    build_analysis_text_prompt,
    build_user_message_content,
)
from src.services.ai_response_parser import (
    ensure_response_is_complete,
    extract_ai_response_content,
    parse_ai_response_json,
)
from src.services.ai_request_compat import (
    CHAT_COMPLETIONS_API_MODE,
    RESPONSES_API_MODE,
    build_ai_request_params,
    create_ai_response_async,
    is_chat_completions_api_unsupported_error,
    is_json_output_unsupported_error,
    is_responses_api_unsupported_error,
    is_temperature_unsupported_error,
    is_thinking_unsupported_error,
    normalize_thinking_param_style,
    remove_temperature_param,
    remove_thinking_param,
)
from src.services.notification_service import build_notification_service
from src.services.ai_retry_policy import (
    AIRetryPolicy,
    format_retry_note,
    is_output_budget_error,
)
from src.services.image_type_detector import (
    MIN_IMAGE_BYTES,
    detect_image_mime,
    is_placeholder_image,
)
from src.utils import convert_goofish_link, retry_on_failure


def _positive_int(value, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


DEFAULT_IMAGE_DOWNLOAD_CONCURRENCY = max(
    1,
    _positive_int(os.getenv("IMAGE_DOWNLOAD_CONCURRENCY", "3"), 3),
)


def safe_print(text):
    """安全的打印函数，处理编码错误"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果遇到编码错误，尝试用ASCII编码并忽略无法编码的字符
        try:
            print(text.encode('ascii', errors='ignore').decode('ascii'))
        except:
            # 如果还是失败，打印一个简化的消息
            print("[输出包含无法显示的字符]")


def _build_debug_request_summary(api_mode: str, request_params: dict) -> dict:
    summary = {
        "api_mode": api_mode,
        "model": request_params.get("model"),
    }
    if "temperature" in request_params:
        summary["temperature"] = request_params["temperature"]
    if "max_output_tokens" in request_params:
        summary["max_output_tokens"] = request_params["max_output_tokens"]
    if "max_tokens" in request_params:
        summary["max_tokens"] = request_params["max_tokens"]
    if "text" in request_params:
        summary["text"] = request_params["text"]
    if "response_format" in request_params:
        summary["response_format"] = request_params["response_format"]
    if "input" in request_params:
        summary["input_content_types"] = [
            [item.get("type") for item in message.get("content", [])]
            for message in request_params["input"]
        ]
    if "messages" in request_params:
        summary["message_content_types"] = [
            _extract_message_content_types(message)
            for message in request_params["messages"]
        ]
    return summary


def _extract_message_content_types(message: dict) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return ["text"]
    if not isinstance(content, list):
        return [type(content).__name__]
    return [str(item.get("type")) for item in content if isinstance(item, dict)]


@retry_on_failure(retries=2, delay=3)
async def _download_single_image(url, save_path):
    """下载单个图片，校验内容类型与文件头，写临时文件后原子替换。"""
    loop = asyncio.get_running_loop()
    # 使用 run_in_executor 运行同步的 requests 代码，避免阻塞事件循环
    response = await loop.run_in_executor(
        None,
        lambda: requests.get(url, headers=IMAGE_DOWNLOAD_HEADERS, timeout=20, stream=True)
    )
    try:
        response.raise_for_status()
        content_type = str(response.headers.get("Content-Type", "")).split(";")[0]
        head = next(response.iter_content(chunk_size=1024), b"")
        if not content_type.startswith("image/"):
            safe_print(
                f"   [图片] 响应不是图片，跳过下载: {url} "
                f"(Content-Type: {content_type or '未知'}, {len(head)} 字节)"
            )
            return None
        if is_placeholder_image(head):
            safe_print(f"   [图片] 收到 1x1 占位图，跳过下载: {url}")
            return None

        part_path = f"{save_path}.part"
        written = len(head)
        with open(part_path, 'wb') as f:
            f.write(head)
            for chunk in response.iter_content(chunk_size=8192):
                written += len(chunk)
                f.write(chunk)
        if written < MIN_IMAGE_BYTES:
            os.remove(part_path)
            safe_print(f"   [图片] 图片过小（{written} 字节），跳过: {url}")
            return None
        os.replace(part_path, save_path)
    finally:
        response.close()
    return save_path


def _build_image_save_path(
    product_id: str,
    index: int,
    url: str,
    task_image_dir: str,
) -> str:
    clean_url = url.split('.heic')[0] if '.heic' in url else url
    file_name_base = os.path.basename(clean_url).split('?')[0]
    file_name = f"product_{product_id}_{index}_{file_name_base}"
    file_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
    if not os.path.splitext(file_name)[1]:
        file_name += ".jpg"
    return os.path.join(task_image_dir, file_name)


async def download_all_images(product_id, image_urls, task_name="default", concurrency=None):
    """异步下载一个商品的所有图片。如果图片已存在则跳过。支持任务隔离。"""
    if not image_urls:
        return []

    # 为每个任务创建独立的图片目录
    task_image_dir = os.path.join(IMAGE_SAVE_DIR, f"{TASK_IMAGE_DIR_PREFIX}{task_name}")
    os.makedirs(task_image_dir, exist_ok=True)

    urls = [url.strip() for url in image_urls if url.strip().startswith('http')]
    if not urls:
        return []

    max_concurrency = _positive_int(concurrency, DEFAULT_IMAGE_DOWNLOAD_CONCURRENCY)
    semaphore = asyncio.Semaphore(max_concurrency)
    total_images = len(urls)

    async def _download_one(index: int, url: str):
        save_path = _build_image_save_path(product_id, index, url, task_image_dir)
        if os.path.exists(save_path):
            safe_print(
                f"   [图片] 图片 {index}/{total_images} 已存在，跳过下载: {os.path.basename(save_path)}"
            )
            return save_path
        async with semaphore:
            safe_print(f"   [图片] 正在下载图片 {index}/{total_images}: {url}")
            if await _download_single_image(url, save_path):
                safe_print(
                    f"   [图片] 图片 {index}/{total_images} 已成功下载到: {os.path.basename(save_path)}"
                )
                return save_path
        return None

    tasks = [
        asyncio.create_task(_download_one(index, url))
        for index, url in enumerate(urls, start=1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    saved_paths = []
    for index, (url, result) in enumerate(zip(urls, results), start=1):
        try:
            if isinstance(result, Exception):
                raise result
            if result:
                saved_paths.append(result)
            else:
                safe_print(f"   [图片] 图片 {index}/{total_images} 未能保存: {url}")
        except Exception as e:
            safe_print(f"   [图片] 处理图片 {url} 时发生错误，已跳过此图: {e}")

    return saved_paths


def cleanup_task_images(task_name):
    """清理指定任务的图片目录"""
    task_image_dir = os.path.join(IMAGE_SAVE_DIR, f"{TASK_IMAGE_DIR_PREFIX}{task_name}")
    if os.path.exists(task_image_dir):
        try:
            shutil.rmtree(task_image_dir)
            safe_print(f"   [清理] 已删除任务 '{task_name}' 的临时图片目录: {task_image_dir}")
        except Exception as e:
            safe_print(f"   [清理] 删除任务 '{task_name}' 的临时图片目录时出错: {e}")
    else:
        safe_print(f"   [清理] 任务 '{task_name}' 的临时图片目录不存在: {task_image_dir}")


def cleanup_ai_logs(logs_dir: str, keep_days: int = 1) -> None:
    try:
        cutoff = datetime.now() - timedelta(days=keep_days)
        for filename in os.listdir(logs_dir):
            if not filename.endswith(".log"):
                continue
            try:
                timestamp = datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            if timestamp < cutoff:
                os.remove(os.path.join(logs_dir, filename))
    except Exception as e:
        safe_print(f"   [日志] 清理AI日志时出错: {e}")


def encode_image_to_base64(image_path):
    """将本地图片文件编码为 Base64 字符串。"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        safe_print(f"编码图片时出错: {e}")
        return None


def validate_ai_response_format(parsed_response):
    """验证AI响应的格式是否符合预期结构"""
    required_fields = [
        "prompt_version",
        "is_recommended",
        "reason",
        "risk_tags",
        "criteria_analysis"
    ]

    # 检查顶层字段
    for field in required_fields:
        if field not in parsed_response:
            safe_print(f"   [AI分析] 警告：响应缺少必需字段 '{field}'")
            return False

    # 检查criteria_analysis是否为字典且不为空
    criteria_analysis = parsed_response.get("criteria_analysis", {})
    if not isinstance(criteria_analysis, dict) or not criteria_analysis:
        safe_print("   [AI分析] 警告：criteria_analysis必须是非空字典")
        return False

    # 检查seller_type字段（所有商品都需要）
    if "seller_type" not in criteria_analysis:
        safe_print("   [AI分析] 警告：criteria_analysis缺少必需字段 'seller_type'")
        return False

    # 检查数据类型
    if not isinstance(parsed_response.get("is_recommended"), bool):
        safe_print("   [AI分析] 警告：is_recommended字段不是布尔类型")
        return False

    if not isinstance(parsed_response.get("risk_tags"), list):
        safe_print("   [AI分析] 警告：risk_tags字段不是列表类型")
        return False

    return True


@retry_on_failure(retries=3, delay=5)
async def send_ntfy_notification(product_data, reason):
    """兼容旧调用名，内部统一走 NotificationService。"""
    service = build_notification_service()
    if not service.clients:
        safe_print(
            "警告：未在 .env 文件中配置任何通知服务，跳过通知。"
        )
        return {}

    results = await service.send_notification(product_data, reason)
    for channel, result in results.items():
        if result["success"]:
            safe_print(f"   -> {channel} 通知发送成功。")
            continue
        safe_print(f"   -> {channel} 通知发送失败: {result['message']}")
    return results


async def get_ai_analysis(product_data, image_paths=None, prompt_text=""):
    """将完整的商品JSON数据和所有图片发送给 AI 进行分析（异步）。"""
    if not client:
        safe_print("   [AI分析] 错误：AI客户端未初始化，跳过分析。")
        return None

    item_info = product_data.get('商品信息', {})
    product_id = item_info.get('商品ID', 'N/A')

    safe_print(f"\n   [AI分析] 开始分析商品 #{product_id} (含 {len(image_paths or [])} 张图片)...")
    safe_print(f"   [AI分析] 标题: {item_info.get('商品标题', '无')}")

    if not prompt_text:
        safe_print("   [AI分析] 错误：未提供AI分析所需的prompt文本。")
        return None

    product_details_json = json.dumps(product_data, ensure_ascii=False, indent=2)
    system_prompt = prompt_text

    if AI_DEBUG_MODE:
        safe_print("\n--- [AI DEBUG] ---")
        safe_print("--- PRODUCT DATA (JSON) ---")
        safe_print(product_details_json)
        safe_print("--- PROMPT TEXT (完整内容) ---")
        safe_print(prompt_text)
        safe_print("-------------------\n")

    image_data_urls = []
    if image_paths:
        for path in image_paths:
            base64_image = encode_image_to_base64(path)
            if base64_image:
                image_data_urls.append(
                    f"data:{detect_image_mime(path)};base64,{base64_image}"
                )

    combined_text_prompt = build_analysis_text_prompt(
        product_details_json,
        system_prompt,
        include_images=bool(image_data_urls),
    )
    user_content = build_user_message_content(combined_text_prompt, image_data_urls)
    messages = [{"role": "user", "content": user_content}]

    # 保存最终传输内容到日志文件
    try:
        # 创建logs文件夹
        logs_dir = os.path.join("logs", "ai")
        os.makedirs(logs_dir, exist_ok=True)
        cleanup_ai_logs(logs_dir, keep_days=1)

        # 生成日志文件名（当前时间）
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{current_time}.log"
        log_filepath = os.path.join(logs_dir, log_filename)

        task_name = product_data.get("任务名称") or product_data.get("任务名") or "unknown"
        log_payload = {
            "timestamp": current_time,
            "task_name": task_name,
            "product_id": product_id,
            "title": item_info.get("商品标题", "无"),
            "image_count": len(image_data_urls),
        }
        log_content = json.dumps(log_payload, ensure_ascii=False)

        # 写入日志文件
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write(log_content)

        safe_print(f"   [日志] AI分析请求已保存到: {log_filepath}")

    except Exception as e:
        safe_print(f"   [日志] 保存AI分析日志时出错: {e}")

    # AI 调用：应用层统一负责尝试计数、退避节奏与输出预算递增
    policy = AIRetryPolicy.from_settings()
    log_scope = f"[AI分析] 商品 #{product_id}"
    api_mode = CHAT_COMPLETIONS_API_MODE
    use_response_format = ENABLE_RESPONSE_FORMAT
    use_temperature = True
    thinking_param_style = normalize_thinking_param_style(AI_THINKING_PARAM_STYLE)
    reasoning_effort = str(AI_REASONING_EFFORT or "").strip()
    use_reasoning_control = ENABLE_THINKING or bool(reasoning_effort)
    max_output_tokens = policy.base_output_tokens

    for attempt in policy.attempts():
        try:
            # 根据重试次数调整参数
            current_temperature = 0.1 if attempt == 1 else 0.05  # 重试时使用更低的温度

            from src.config import get_ai_request_params

            request_params = build_ai_request_params(
                api_mode,
                model=MODEL_NAME,
                messages=messages,
                temperature=current_temperature,
                max_output_tokens=max_output_tokens,
                enable_json_output=use_response_format,
            )
            if not use_temperature:
                request_params = remove_temperature_param(request_params)

            request_params = get_ai_request_params(api_mode=api_mode, **request_params)
            if not use_reasoning_control:
                request_params = remove_thinking_param(request_params)

            if AI_DEBUG_MODE:
                safe_print(f"\n--- [AI DEBUG] {log_scope} 第{attempt}次尝试 REQUEST ---")
                safe_print(
                    json.dumps(
                        _build_debug_request_summary(api_mode, request_params),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                safe_print("-----------------------------------\n")

            response = await create_ai_response_async(
                client,
                api_mode,
                request_params,
            )
            ensure_response_is_complete(response)
            ai_response_content = extract_ai_response_content(response)

            if AI_DEBUG_MODE:
                safe_print(f"\n--- [AI DEBUG] {log_scope} 第{attempt}次尝试 ---")
                safe_print("--- RAW AI RESPONSE ---")
                safe_print(ai_response_content)
                safe_print("---------------------\n")

            parsed_response = parse_ai_response_json(ai_response_content)
            if not validate_ai_response_format(parsed_response):
                raise ValueError("AI响应格式缺少必需字段或字段类型不正确。")

            safe_print(f"   {log_scope} {policy.describe_attempt(attempt)}成功，响应格式验证通过")
            return parsed_response

        except Exception as e:
            if (
                api_mode == CHAT_COMPLETIONS_API_MODE
                and is_chat_completions_api_unsupported_error(e)
            ):
                api_mode = RESPONSES_API_MODE
                safe_print(
                    "   [AI分析] 当前服务未实现 Chat Completions API，后续重试将自动回退到 Responses API。"
                )
            elif api_mode == RESPONSES_API_MODE and is_responses_api_unsupported_error(e):
                api_mode = CHAT_COMPLETIONS_API_MODE
                safe_print(
                    "   [AI分析] 当前服务未实现 Responses API，后续重试将自动回退到 Chat Completions API。"
                )
            if use_response_format and is_json_output_unsupported_error(e):
                use_response_format = False
                safe_print(
                    "   [AI分析] 当前模型不支持结构化 JSON 输出，后续重试将自动禁用该参数。"
                )
            if use_temperature and is_temperature_unsupported_error(e):
                use_temperature = False
                safe_print(
                    "   [AI分析] 当前模型不支持 temperature 参数，后续重试将自动禁用该参数。"
                )
            if use_reasoning_control and is_thinking_unsupported_error(e):
                use_reasoning_control = False
                safe_print(
                    "   [AI分析] 当前模型不支持思考控制参数，后续重试将自动移除该参数。"
                )
            if is_output_budget_error(e) and max_output_tokens < policy.output_token_cap:
                escalated_tokens = policy.escalate_output_tokens(max_output_tokens)
                if escalated_tokens > max_output_tokens:
                    max_output_tokens = escalated_tokens
                    safe_print(
                        f"   {log_scope} 输出预算不足，下一次尝试将提高到 {max_output_tokens} tokens"
                    )

            if AI_DEBUG_MODE:
                safe_print(f"\n--- [AI DEBUG] {log_scope} 第{attempt}次尝试 EXCEPTION ---")
                safe_print(repr(e))
                safe_print(traceback.format_exc())
                safe_print("-------------------------------------\n")
            safe_print(f"   {log_scope} {policy.describe_attempt(attempt)}失败: {e}")
            if not policy.has_next_attempt(attempt):
                raise

            delay = policy.delay_seconds(attempt, e)
            safe_print(f"   {log_scope} {format_retry_note(policy, attempt, delay)}")
            if delay > 0:
                await asyncio.sleep(delay)
