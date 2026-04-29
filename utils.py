"""
工具函数模块 — 通用辅助功能
"""
from __future__ import annotations

import json
import re
import uuid
import asyncio
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger

PLUGIN_ID = "astrbot_plugin_novel"


def generate_id(prefix: str = "") -> str:
    """生成唯一 ID，如 idea_a1b2c3d4"""
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{short}" if prefix else short


def safe_json_load(path: Path, default: Any = None) -> Any:
    """安全加载 JSON 文件，文件不存在或解析失败时返回 default"""
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[{PLUGIN_ID}] JSON 加载失败 {path}: {e}")
        return default


def safe_json_save(path: Path, data: Any) -> None:
    """安全写入 JSON（先写临时文件再 rename，防止写到一半崩溃导致数据损坏）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as e:
        logger.error(f"[{PLUGIN_ID}] JSON 保存失败 {path}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def truncate_text(text: str, max_len: int = 500) -> str:
    """截断文本，超出部分用省略号代替"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def call_llm(provider, prompt: str, system_prompt: str = "", timeout: int = 120) -> str:
    """
    封装 LLM 调用，含超时和错误处理。
    provider: AstrBot 的 LLM provider 实例
    返回: AI 生成的文本
    """
    try:
        result = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                system_prompt=system_prompt or None,
            ),
            timeout=timeout,
        )
        # AstrBot provider 返回的可能是字符串或对象
        if hasattr(result, "completion_text"):
            return result.completion_text or ""
        if isinstance(result, str):
            return result
        return str(result)
    except asyncio.TimeoutError:
        logger.error(f"[{PLUGIN_ID}] LLM 调用超时 ({timeout}s)")
        raise
    except Exception as e:
        logger.error(f"[{PLUGIN_ID}] LLM 调用失败: {e}")
        raise


def parse_json_from_response(text: str) -> Optional[dict]:
    """
    从 AI 响应中提取 JSON 对象。
    支持 ```json ... ``` 代码块格式和裸 JSON。
    """
    # 尝试从代码块中提取
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试直接寻找 { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"[{PLUGIN_ID}] 无法从 AI 响应中解析 JSON")
    return None


def format_timestamp() -> str:
    """返回当前时间的 ISO 格式字符串"""
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# =====================================================================
# 封面图片生成（支持 OpenAI images API + Gemini 原生 API）
# =====================================================================

async def generate_cover_image(
    provider,
    prompt: str,
    output_path: Path,
    size: str = "1024x1536",
    reference_image_path: Optional[Path] = None,
    timeout: int = 180,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    responses_model: str = "",
) -> Optional[Path]:
    """
    生成封面图片，支持多种 API 协议。
    尝试顺序：OpenAI Images HTTP API → OpenAI SDK images API → Gemini 原生 API
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试获取 provider 底层 client
    client = None
    if hasattr(provider, "client"):
        client = provider.client
    elif hasattr(provider, "_client"):
        client = provider._client

    # 模型名：优先使用配置指定的，其次从 provider 自动获取
    if not model_name:
        for attr in ("model_name", "model", "model_id", "_model_name", "_model"):
            if hasattr(provider, attr):
                val = getattr(provider, attr)
                if isinstance(val, str) and val:
                    model_name = val
                    break
    if model_name:
        logger.info(f"[{PLUGIN_ID}] 绘图模型：{model_name}")

    direct_api_key, direct_base_url = _extract_openai_image_api_config(
        provider,
        api_key=api_key,
        base_url=base_url,
    )
    direct_requested = bool(api_key or base_url) or _is_gpt_image_model(model_name)

    try:
        # ---- 方式 1: OpenAI Images HTTP API ----
        # gpt-image-2 只能走 /v1/images/generations 或 /v1/images/edits。
        # 有些 AstrBot provider 会把它当普通聊天模型路由，这里直接请求 Images API。
        if direct_requested and direct_api_key:
            logger.info(f"[{PLUGIN_ID}] 尝试 OpenAI Images HTTP API 生成封面...")
            try:
                response = await _call_openai_images_http(
                    api_key=direct_api_key,
                    base_url=direct_base_url,
                    prompt=prompt,
                    reference_path=reference_image_path,
                    size=size,
                    model_name=model_name or "gpt-image-2",
                    responses_model=responses_model,
                    timeout=timeout,
                )
                if _save_openai_image_response(response, output_path):
                    logger.info(f"[{PLUGIN_ID}] 封面生成完成（OpenAI Images HTTP）：{output_path}")
                    return output_path
                logger.warning(
                    f"[{PLUGIN_ID}] OpenAI Images HTTP API 返回成功，"
                    f"但响应中未找到可保存的图片，尝试 SDK/Gemini"
                )
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] OpenAI Images HTTP API 失败: {e}，尝试 SDK/Gemini")

        if client is None:
            logger.error(f"[{PLUGIN_ID}] 封面生成失败：无法获取 provider 底层 client")
            return None

        # ---- 方式 2: OpenAI 兼容 SDK images API ----
        if hasattr(client, "images"):
            logger.info(f"[{PLUGIN_ID}] 尝试 OpenAI SDK images API 生成封面...")
            try:
                if reference_image_path and reference_image_path.exists():
                    response = await asyncio.wait_for(
                        _call_images_edit(client, prompt, reference_image_path, size, model_name),
                        timeout=timeout,
                    )
                else:
                    response = await asyncio.wait_for(
                        _call_images_generate(client, prompt, size, model_name),
                        timeout=timeout,
                    )

                if _save_openai_image_response(response, output_path):
                    logger.info(f"[{PLUGIN_ID}] 封面生成完成（OpenAI SDK）：{output_path}")
                    return output_path
            except AttributeError:
                logger.info(f"[{PLUGIN_ID}] OpenAI images API 不可用，尝试 Gemini")
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] OpenAI images API 失败: {e}，尝试 Gemini")

        # ---- 方式 3: Gemini 原生 generate_content API ----
        logger.info(f"[{PLUGIN_ID}] 尝试 Gemini 原生 API 生成封面...")
        img_bytes = await asyncio.wait_for(
            _call_gemini_image(client, provider, prompt, reference_image_path),
            timeout=timeout,
        )
        if img_bytes:
            output_path.write_bytes(img_bytes)
            logger.info(f"[{PLUGIN_ID}] 封面生成完成（Gemini API）：{output_path}")
            return output_path

        logger.error(f"[{PLUGIN_ID}] 封面生成失败：所有 API 均未返回有效图片")
        return None

    except asyncio.TimeoutError:
        logger.error(f"[{PLUGIN_ID}] 封面生成超时（{timeout}s）")
        return None
    except Exception as e:
        logger.error(f"[{PLUGIN_ID}] 封面生成失败: {e}")
        return None


async def _call_gemini_image(
    client, provider, prompt: str, reference_image_path: Optional[Path] = None
) -> Optional[bytes]:
    """使用 Gemini 原生 generate_content API 生成图片"""
    import inspect

    # 获取模型名称
    model_name = ""
    for attr in ("model_name", "model", "model_id", "_model_name", "_model"):
        if hasattr(provider, attr):
            val = getattr(provider, attr)
            if isinstance(val, str) and val:
                model_name = val
                break

    # ---- 尝试 google-genai SDK (新版 google.genai.Client) ----
    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )

            # 构建请求内容（支持图生图：参考图作为 Part 传入）
            if reference_image_path and reference_image_path.exists():
                ref_bytes = reference_image_path.read_bytes()
                ext = reference_image_path.suffix.lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                parts = [
                    types.Part.from_text(
                        f"请参考以下图片的画风和色彩风格来生成封面。\n\n{prompt}"
                    ),
                    types.Part.from_bytes(data=ref_bytes, mime_type=mime),
                ]
                req_contents = [types.Content(parts=parts)]
            else:
                req_contents = prompt

            result = client.models.generate_content(
                model=model_name or "gemini-2.0-flash-exp",
                contents=req_contents,
                config=config,
            )
            if inspect.isawaitable(result):
                result = await result

            img = _extract_gemini_image(result)
            if img:
                return img
        except ImportError:
            logger.warning(f"[{PLUGIN_ID}] google-genai SDK 未安装")
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] Gemini google-genai API 失败: {e}")

    # ---- 尝试 google-generativeai SDK (旧版 GenerativeModel) ----
    if hasattr(client, "generate_content"):
        try:
            result = client.generate_content(prompt)
            if inspect.isawaitable(result):
                result = await result

            img = _extract_gemini_image(result)
            if img:
                return img
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] Gemini GenerativeModel API 失败: {e}")

    return None


def _extract_gemini_image(result) -> Optional[bytes]:
    """从 Gemini API 返回结果中提取图片字节"""
    import base64

    if not result:
        return None

    # google-genai SDK 格式: result.candidates[].content.parts[].inline_data
    if hasattr(result, "candidates") and result.candidates:
        for candidate in result.candidates:
            if not hasattr(candidate, "content") or not candidate.content:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        return base64.b64decode(data)
                    if isinstance(data, bytes):
                        return data

    # google-generativeai SDK 格式: result.parts[].inline_data
    if hasattr(result, "parts"):
        for part in result.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                data = part.inline_data.data
                if isinstance(data, str):
                    return base64.b64decode(data)
                if isinstance(data, bytes):
                    return data

    return None


# =====================================================================
# OpenAI 兼容 API 辅助函数
# =====================================================================

def _is_gpt_image_model(model_name: str) -> bool:
    """判断是否是 GPT Image 系列模型。"""
    normalized = (model_name or "").strip().lower().replace("_", "-")
    return "gpt-image" in normalized


def _normalize_openai_image_model_name(model_name: str) -> str:
    """把 provider id 中的模型名部分提取出来，例如 openai_1/gpt-image-2 -> gpt-image-2。"""
    text = (model_name or "").strip()
    if "/" in text:
        last = text.rsplit("/", 1)[-1].strip()
        if _is_gpt_image_model(last):
            return last
    return text


def _stringify_secret(value) -> str:
    """把 provider/client 中的配置值安全转成字符串。"""
    if value is None:
        return ""
    try:
        if hasattr(value, "get_secret_value"):
            value = value.get_secret_value()
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in ("none", "null"):
        return ""
    return text


def _extract_config_value(obj, keys: tuple[str, ...]) -> str:
    """从对象属性或内嵌 config dict 中提取配置值。"""
    if not obj:
        return ""

    for key in keys:
        try:
            if hasattr(obj, key):
                val = _stringify_secret(getattr(obj, key))
                if val:
                    return val
        except Exception:
            pass

    for cfg_name in ("config", "_config", "provider_config", "provider_settings"):
        try:
            cfg = getattr(obj, cfg_name, None)
        except Exception:
            cfg = None
        if not cfg:
            continue
        for key in keys:
            try:
                if isinstance(cfg, dict):
                    val = cfg.get(key)
                elif hasattr(cfg, "get"):
                    val = cfg.get(key)
                else:
                    val = getattr(cfg, key, None)
                val = _stringify_secret(val)
                if val:
                    return val
            except Exception:
                pass

    return ""


def _normalize_openai_images_base_url(base_url: str) -> str:
    """规范化 OpenAI 兼容 base_url，返回不带结尾斜杠的 /v1 地址。"""
    base = (base_url or "").strip() or "https://api.openai.com/v1"
    base = base.rstrip("/")
    for suffix in (
        "/images/generations",
        "/images/edits",
        "/chat/completions",
        "/responses",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/")


def _extract_openai_image_api_config(provider, api_key: str = "", base_url: str = "") -> tuple[str, str]:
    """从显式配置、provider、SDK client 或环境变量中提取 OpenAI Images API 配置。"""
    import os

    api_key = _stringify_secret(api_key)
    base_url = _stringify_secret(base_url)

    candidates = []
    if provider:
        candidates.append(provider)
        try:
            if hasattr(provider, "client"):
                candidates.append(provider.client)
        except Exception:
            pass
        try:
            if hasattr(provider, "_client"):
                candidates.append(provider._client)
        except Exception:
            pass

    if not api_key:
        key_names = ("api_key", "_api_key", "openai_api_key", "key", "token", "access_token")
        for obj in candidates:
            api_key = _extract_config_value(obj, key_names)
            if api_key:
                break

    if not base_url:
        base_names = (
            "base_url",
            "_base_url",
            "api_base",
            "api_base_url",
            "openai_api_base",
            "endpoint",
        )
        for obj in candidates:
            base_url = _extract_config_value(obj, base_names)
            if base_url:
                break

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    return api_key, _normalize_openai_images_base_url(base_url)


def _save_openai_image_response(response, output_path: Path) -> bool:
    """保存 OpenAI Images API/SDK 的返回图片。"""
    import base64
    import urllib.request

    if not response:
        return False

    data = None
    if isinstance(response, dict):
        data = response.get("data")
    elif hasattr(response, "data"):
        data = response.data

    if not data:
        return _save_responses_image_output(response, output_path)

    first = data[0]
    b64_json = ""
    image_url = ""
    if isinstance(first, dict):
        b64_json = first.get("b64_json", "")
        image_url = first.get("url", "")
    else:
        b64_json = getattr(first, "b64_json", "")
        image_url = getattr(first, "url", "")

    if b64_json:
        output_path.write_bytes(base64.b64decode(b64_json))
        return True
    if image_url:
        urllib.request.urlretrieve(image_url, str(output_path))
        return True
    return False


def _save_responses_image_output(response, output_path: Path) -> bool:
    """保存 Responses API image_generation_call 风格的返回图片。"""
    import base64
    import urllib.request

    outputs = []
    if isinstance(response, dict):
        outputs = response.get("output") or []
    elif hasattr(response, "output"):
        outputs = response.output or []

    for item in outputs:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", "")
        if item_type != "image_generation_call":
            continue
        b64_data = item.get("result", "") if isinstance(item, dict) else getattr(item, "result", "")
        if isinstance(b64_data, dict):
            image_url = b64_data.get("url") or b64_data.get("image_url")
            b64_data = (
                b64_data.get("b64_json")
                or b64_data.get("image_base64")
                or b64_data.get("base64")
                or b64_data.get("data")
                or ""
            )
            if image_url:
                urllib.request.urlretrieve(image_url, str(output_path))
                return True
        if isinstance(b64_data, str) and b64_data.startswith("data:image/"):
            b64_data = b64_data.split(",", 1)[-1]
        if b64_data:
            output_path.write_bytes(base64.b64decode(b64_data))
            return True
    return False


async def _call_openai_images_http(
    api_key: str,
    base_url: str,
    prompt: str,
    reference_path: Optional[Path],
    size: str,
    model_name: str,
    responses_model: str,
    timeout: int,
) -> dict:
    """直接调用 OpenAI Images HTTP API，支持 generations 和 edits。"""
    import urllib.error
    import urllib.request

    endpoint = "images/edits" if reference_path and reference_path.exists() else "images/generations"
    url = f"{_normalize_openai_images_base_url(base_url)}/{endpoint}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "AstrBot-Novel-Plugin/1.0",
    }

    def _post_with_body(body: bytes, req_headers: dict):
        req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")

        def _post():
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI Images API HTTP {e.code}: {detail}") from e

        return _post

    if endpoint == "images/generations":
        request_model = _normalize_openai_image_model_name(model_name) or "gpt-image-2"
        payload = {
            "model": request_model,
            "prompt": prompt,
            "size": size,
        }
        if not _is_gpt_image_model(request_model):
            payload["n"] = 1
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        loop = asyncio.get_event_loop()
        try:
            raw = await loop.run_in_executor(None, _post_with_body(body, headers))
        except RuntimeError as e:
            if not _is_missing_image_generation_tool_error(str(e)):
                raise
            logger.warning(
                f"[{PLUGIN_ID}] 网关的 Images 端点要求 Responses image_generation 工具参数，"
                f"正在改用 /v1/responses 兼容模式重试..."
            )
            return await _call_openai_responses_image_generation_http(
                api_key=api_key,
                base_url=base_url,
                prompt=prompt,
                reference_path=reference_path,
                size=size,
                model_name=_select_responses_image_model(model_name, responses_model),
                timeout=timeout,
            )
        return json.loads(raw)
    else:
        request_model = _normalize_openai_image_model_name(model_name) or "gpt-image-2"
        fields = {
            "model": request_model,
            "prompt": prompt,
            "size": size,
        }
        if not _is_gpt_image_model(request_model):
            fields["n"] = "1"
        body, content_type = _build_multipart_form_data(fields, "image", reference_path)
        headers["Content-Type"] = content_type
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _post_with_body(body, headers))
        return json.loads(raw)


def _is_missing_image_generation_tool_error(error_text: str) -> bool:
    """识别部分 OpenAI 兼容网关把 Images 请求转成 Responses 工具调用时的缺参错误。"""
    text = (error_text or "").lower()
    return (
        "image_generation" in text
        and "tool_choice" in text
        and "tools" in text
        and "not found" in text
    )


def _select_responses_image_model(image_model_name: str, responses_model: str = "") -> str:
    """Responses image_generation 工具需要主模型；未配置时给一个官方示例可用默认值。"""
    configured = (responses_model or "").strip()
    if configured:
        return configured.rsplit("/", 1)[-1].strip()
    normalized = _normalize_openai_image_model_name(image_model_name)
    if normalized and not _is_gpt_image_model(normalized):
        return normalized
    return "gpt-5"


async def _call_openai_responses_image_generation_http(
    api_key: str,
    base_url: str,
    prompt: str,
    reference_path: Optional[Path],
    size: str,
    model_name: str,
    timeout: int,
) -> dict:
    """调用 /v1/responses + image_generation 工具，兼容部分网关的 GPT Image 路由。"""
    import urllib.error
    import urllib.request

    url = f"{_normalize_openai_images_base_url(base_url)}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "AstrBot-Novel-Plugin/1.0",
    }
    tool = {"type": "image_generation"}
    payload = {
        "model": model_name,
        "input": _build_responses_image_input(prompt, reference_path),
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
    }

    async def _post_payload(data: dict) -> dict:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        loop = asyncio.get_event_loop()

        def _post():
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI Responses API HTTP {e.code}: {detail}") from e

        raw = await loop.run_in_executor(None, _post)
        return json.loads(raw)

    try:
        return await _post_payload(payload)
    except RuntimeError as e:
        if not _is_missing_image_generation_tool_error(str(e)):
            raise
        logger.warning(
            f"[{PLUGIN_ID}] 网关不接受对象格式 tool_choice，"
            f"正在改用字符串格式重试..."
        )

    payload["tool_choice"] = "image_generation"
    try:
        return await _post_payload(payload)
    except RuntimeError as e:
        if not _is_missing_image_generation_tool_error(str(e)):
            raise
        logger.warning(
            f"[{PLUGIN_ID}] 网关不接受强制 image_generation tool_choice，"
            f"正在移除 tool_choice 后重试..."
        )

    payload.pop("tool_choice", None)
    return await _post_payload(payload)


def _build_responses_image_input(prompt: str, reference_path: Optional[Path]):
    """构建 Responses API 输入；有参考图时把图片以内联 data URL 传入。"""
    if not reference_path or not reference_path.exists():
        return prompt
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _path_to_data_url(reference_path)},
        ],
    }]


def _path_to_data_url(path: Path) -> str:
    """把本地图片转成 data URL。"""
    import base64
    import mimetypes

    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_multipart_form_data(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """构建 multipart/form-data 请求体，避免新增 requests 依赖。"""
    import mimetypes

    boundary = f"----astrbot-novel-{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    mime = mimetypes.guess_type(str(file_path))[0] or "image/png"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), f"multipart/form-data; boundary={boundary}"

async def _call_images_edit(client, prompt: str, reference_path: Path, size: str, model_name: str = ""):
    """图生图模式：使用 OpenAI images.edit() API"""
    import inspect

    if hasattr(client, "images") and hasattr(client.images, "edit"):
        try:
            with open(reference_path, "rb") as img_file:
                kwargs = {
                    "image": img_file,
                    "prompt": prompt,
                    "size": size,
                }
                if model_name:
                    kwargs["model"] = _normalize_openai_image_model_name(model_name)
                if not _is_gpt_image_model(model_name):
                    kwargs["n"] = 1
                    kwargs["response_format"] = "b64_json"
                result = client.images.edit(**kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] images.edit() 失败: {e}")

    # 回退到 images.generate
    return await _call_images_generate(client, prompt, size, model_name)


async def _call_images_generate(client, prompt: str, size: str, model_name: str = ""):
    """OpenAI images.generate 调用"""
    import inspect

    if hasattr(client, "images") and hasattr(client.images, "generate"):
        kwargs = {
            "prompt": prompt,
            "size": size,
        }
        if model_name:
            kwargs["model"] = _normalize_openai_image_model_name(model_name)
        if not _is_gpt_image_model(model_name):
            kwargs["n"] = 1
            kwargs["response_format"] = "b64_json"
        result = client.images.generate(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    raise AttributeError("client 不支持 images.generate()")


# =====================================================================
# Mermaid 图表渲染为图片
# =====================================================================

async def render_mermaid_to_image(
    mermaid_code: str, output_path: Path, timeout: int = 30
) -> Optional[Path]:
    """
    使用 mermaid.ink API 将 Mermaid 代码渲染为 PNG 图片。
    返回图片路径，失败返回 None。
    """
    import base64
    import urllib.request
    import urllib.error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # mermaid.ink API: base64 编码的 mermaid 代码
        encoded = base64.urlsafe_b64encode(
            mermaid_code.encode("utf-8")
        ).decode("ascii")
        url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=!white&theme=neutral"

        # 使用 asyncio 在线程中运行同步下载
        loop = asyncio.get_event_loop()
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "AstrBot-Novel-Plugin/1.0")

        def _download():
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        data = await loop.run_in_executor(None, _download)

        if data and len(data) > 100:  # 确保不是空响应
            output_path.write_bytes(data)
            logger.info(f"[{PLUGIN_ID}] Mermaid 图表渲染完成：{output_path}")
            return output_path
        else:
            logger.warning(f"[{PLUGIN_ID}] Mermaid 渲染返回数据过小")
            return None

    except Exception as e:
        logger.error(f"[{PLUGIN_ID}] Mermaid 图表渲染失败: {e}")
        return None


# =====================================================================
# 图片下载与识别
# =====================================================================

async def download_image(image_url: str, save_dir: Optional[Path] = None) -> Optional[Path]:
    """
    下载图片 URL 到本地临时目录。
    返回本地文件路径，失败返回 None。
    """
    import tempfile
    import urllib.request

    if not image_url:
        return None

    if save_dir is None:
        save_dir = Path(tempfile.gettempdir()) / "astrbot_novel_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 根据 URL 推断扩展名
        ext = ".jpg"
        lower_url = image_url.lower().split("?")[0]
        for e in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            if lower_url.endswith(e):
                ext = e
                break

        filename = f"img_{uuid.uuid4().hex[:12]}{ext}"
        local_path = save_dir / filename

        loop = asyncio.get_event_loop()
        req = urllib.request.Request(image_url)
        req.add_header("User-Agent", "AstrBot-Novel-Plugin/1.0")

        def _dl():
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()

        data = await loop.run_in_executor(None, _dl)
        if data:
            local_path.write_bytes(data)
            return local_path
        return None

    except Exception as e:
        logger.warning(f"[{PLUGIN_ID}] 图片下载失败: {e}")
        return None


async def recognize_image(
    provider,
    image_url: str = "",
    image_path: Optional[Path] = None,
    prompt: str = "",
    timeout: int = 60,
) -> str:
    """
    调用视觉模型识别图片内容。

    尝试顺序：
    1. AstrBot provider 的 text_chat（传入图片 URL）
    2. Google Genai SDK（传入图片字节）
    3. OpenAI 兼容 Chat Completions API（vision 模式）

    参数:
        provider: AstrBot LLM provider 实例（需支持视觉能力）
        image_url: 图片的网络 URL
        image_path: 图片的本地文件路径（优先于 URL）
        prompt: 识别提示词
        timeout: 超时时间（秒）

    返回:
        识别结果文本，失败返回空字符串
    """
    import base64
    import inspect

    if not prompt:
        from .prompts import CHAT_NOVEL_IMAGE_RECOGNITION_PROMPT
        prompt = CHAT_NOVEL_IMAGE_RECOGNITION_PROMPT

    # 准备图片数据
    img_bytes = None
    img_mime = "image/jpeg"
    if image_path and Path(image_path).exists():
        img_bytes = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".gif": "image/gif",
                    ".webp": "image/webp", ".bmp": "image/bmp"}
        img_mime = mime_map.get(ext, "image/jpeg")
    elif image_url:
        # 尝试下载图片
        local = await download_image(image_url)
        if local and local.exists():
            img_bytes = local.read_bytes()
            ext = local.suffix.lower()
            mime_map = {".png": "image/png", ".gif": "image/gif",
                        ".webp": "image/webp", ".bmp": "image/bmp"}
            img_mime = mime_map.get(ext, "image/jpeg")
            # 清理临时文件
            try:
                local.unlink(missing_ok=True)
            except Exception:
                pass

    if not img_bytes and not image_url:
        logger.warning(f"[{PLUGIN_ID}] 图片识别：无图片数据")
        return ""

    img_b64 = base64.b64encode(img_bytes).decode("ascii") if img_bytes else ""

    # ---- 方式 1: AstrBot provider.text_chat 直接传图 ----
    # 部分 provider 支持在 prompt 中内嵌图片 URL 或通过特殊参数传图
    try:
        # 尝试使用 image_url 参数（部分 AstrBot provider 支持）
        result = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                image_urls=[image_url] if image_url else None,
            ),
            timeout=timeout,
        )
        text = ""
        if hasattr(result, "completion_text"):
            text = result.completion_text or ""
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)
        if text and len(text) > 20:
            logger.info(f"[{PLUGIN_ID}] 图片识别成功（AstrBot text_chat）")
            return text
    except TypeError:
        # text_chat 不支持 image_urls 参数，尝试下一种方式
        pass
    except Exception as e:
        logger.debug(f"[{PLUGIN_ID}] AstrBot text_chat 图片识别失败: {e}")

    # ---- 方式 2: Google Genai SDK ----
    if img_bytes:
        client = None
        if hasattr(provider, "client"):
            client = provider.client
        elif hasattr(provider, "_client"):
            client = provider._client

        if client and hasattr(client, "models") and hasattr(client.models, "generate_content"):
            try:
                from google.genai import types

                # 获取模型名
                model_name = ""
                for attr in ("model_name", "model", "model_id", "_model_name", "_model"):
                    if hasattr(provider, attr):
                        val = getattr(provider, attr)
                        if isinstance(val, str) and val:
                            model_name = val
                            break

                parts = [
                    types.Part.from_text(prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
                ]
                req_contents = [types.Content(parts=parts)]

                result = client.models.generate_content(
                    model=model_name or "gemini-2.0-flash",
                    contents=req_contents,
                )
                if inspect.isawaitable(result):
                    result = await result

                # 提取文本
                if hasattr(result, "candidates") and result.candidates:
                    for candidate in result.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    logger.info(f"[{PLUGIN_ID}] 图片识别成功（Genai SDK）")
                                    return part.text
                if hasattr(result, "text") and result.text:
                    logger.info(f"[{PLUGIN_ID}] 图片识别成功（Genai SDK .text）")
                    return result.text
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"[{PLUGIN_ID}] Genai SDK 图片识别失败: {e}")

    # ---- 方式 3: OpenAI 兼容 Chat Completions API (vision) ----
    if img_b64 or image_url:
        client = None
        if hasattr(provider, "client"):
            client = provider.client
        elif hasattr(provider, "_client"):
            client = provider._client

        if client and hasattr(client, "chat") and hasattr(client.chat, "completions"):
            try:
                model_name = ""
                for attr in ("model_name", "model", "model_id", "_model_name", "_model"):
                    if hasattr(provider, attr):
                        val = getattr(provider, attr)
                        if isinstance(val, str) and val:
                            model_name = val
                            break

                # 构建 vision 消息
                image_content = {}
                if img_b64:
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_mime};base64,{img_b64}"
                        }
                    }
                elif image_url:
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content,
                    ]
                }]

                kwargs = {"model": model_name or "gpt-4o", "messages": messages}
                result = client.chat.completions.create(**kwargs)
                if inspect.isawaitable(result):
                    result = await result

                if result and result.choices:
                    text = result.choices[0].message.content or ""
                    if text:
                        logger.info(f"[{PLUGIN_ID}] 图片识别成功（OpenAI Vision API）")
                        return text
            except Exception as e:
                logger.debug(f"[{PLUGIN_ID}] OpenAI Vision API 图片识别失败: {e}")

    logger.warning(f"[{PLUGIN_ID}] 图片识别：所有方式均失败")
    return ""
