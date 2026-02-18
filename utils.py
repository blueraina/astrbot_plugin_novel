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
