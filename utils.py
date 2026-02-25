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
) -> Optional[Path]:
    """
    生成封面图片，支持多种 API 协议。
    尝试顺序：OpenAI images API → Gemini 原生 API
    """
    import base64

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试获取 provider 底层 client
    client = None
    if hasattr(provider, "client"):
        client = provider.client
    elif hasattr(provider, "_client"):
        client = provider._client

    if client is None:
        logger.error(f"[{PLUGIN_ID}] 封面生成失败：无法获取 provider 底层 client")
        return None

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

    try:
        # ---- 方式 1: OpenAI 兼容 images API ----
        if hasattr(client, "images"):
            logger.info(f"[{PLUGIN_ID}] 尝试 OpenAI images API 生成封面...")
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

                if response and response.data:
                    image_data = response.data[0]
                    if hasattr(image_data, "b64_json") and image_data.b64_json:
                        output_path.write_bytes(base64.b64decode(image_data.b64_json))
                        logger.info(f"[{PLUGIN_ID}] 封面生成完成（OpenAI API）：{output_path}")
                        return output_path
                    elif hasattr(image_data, "url") and image_data.url:
                        import urllib.request
                        urllib.request.urlretrieve(image_data.url, str(output_path))
                        logger.info(f"[{PLUGIN_ID}] 封面生成完成（OpenAI URL）：{output_path}")
                        return output_path
            except AttributeError:
                logger.info(f"[{PLUGIN_ID}] OpenAI images API 不可用，尝试 Gemini")
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] OpenAI images API 失败: {e}，尝试 Gemini")

        # ---- 方式 2: Gemini 原生 generate_content API ----
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

async def _call_images_edit(client, prompt: str, reference_path: Path, size: str, model_name: str = ""):
    """图生图模式：使用 OpenAI images.edit() API"""
    import inspect

    if hasattr(client, "images") and hasattr(client.images, "edit"):
        try:
            with open(reference_path, "rb") as img_file:
                kwargs = {
                    "image": img_file,
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "response_format": "b64_json",
                }
                if model_name:
                    kwargs["model"] = model_name
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
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        if model_name:
            kwargs["model"] = model_name
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

