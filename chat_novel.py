"""
群聊小说引擎 — 基于群聊消息自动生成小说
与现有的小说功能完全独立，每个群独立运行。
"""
from __future__ import annotations

import json as _json
import re as _re
from datetime import datetime
from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .utils import (
    PLUGIN_ID,
    generate_id,
    safe_json_load,
    safe_json_save,
    call_llm,
    parse_json_from_response,
    truncate_text,
)
from .prompts import (
    CHAT_NOVEL_GENERATE_CHAPTER_PROMPT,
    CHAT_NOVEL_MAP_CHARACTERS_PROMPT,
    CHAT_NOVEL_EVALUATE_QUALITY_PROMPT,
)


# =====================================================================
# 默认数据模板
# =====================================================================
_DEFAULT_CHAT_NOVEL: dict = {
    "status": "stopped",          # collecting / stopped
    "requirements": "",           # 用户的风格/主题要求
    "title": "群聊物语",
    "chapters": [],
    "characters": [],             # {real_name, novel_name, description, sender_id}
    "global_summary": "",
    "contributors": [],           # 参与聊天的群友昵称列表
    "created_at": "",
}


class ChatNovelEngine:
    """群聊小说引擎 — 收集群聊消息并 AI 生成小说"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._novel_path = data_dir / "chat_novel.json"
        self._messages_path = data_dir / "chat_messages.json"

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def _load_novel(self) -> dict:
        return safe_json_load(self._novel_path, dict(_DEFAULT_CHAT_NOVEL))

    def _save_novel(self, data: dict) -> None:
        safe_json_save(self._novel_path, data)

    def _load_messages(self) -> list:
        raw = safe_json_load(self._messages_path, {"messages": []})
        return raw.get("messages", [])

    def _save_messages(self, messages: list) -> None:
        safe_json_save(self._messages_path, {"messages": messages})

    def is_collecting(self) -> bool:
        novel = self._load_novel()
        return novel.get("status") == "collecting"

    def start(self, requirements: str, title: str = "群聊物语") -> dict:
        """开始收集群聊消息"""
        novel = self._load_novel()
        novel["status"] = "collecting"
        novel["requirements"] = requirements
        novel["title"] = title
        novel["created_at"] = datetime.now().isoformat()
        if not novel.get("global_summary"):
            novel["global_summary"] = "故事尚未开始。"
        self._save_novel(novel)
        # 清空消息缓冲
        self._save_messages([])
        logger.info(f"[{PLUGIN_ID}] 群聊小说开始收集：{title}")
        return novel

    def stop(self) -> None:
        """停止收集"""
        novel = self._load_novel()
        novel["status"] = "stopped"
        self._save_novel(novel)
        logger.info(f"[{PLUGIN_ID}] 群聊小说停止收集")

    def resume(self) -> bool:
        """继续收集（从停止状态恢复，不清空数据）。成功返回 True。"""
        novel = self._load_novel()
        if novel.get("status") == "collecting":
            return False  # 已经在收集中
        if not novel.get("title"):
            return False  # 从未初始化过
        novel["status"] = "collecting"
        self._save_novel(novel)
        logger.info(f"[{PLUGIN_ID}] 群聊小说继续收集")
        return True

    # ------------------------------------------------------------------
    # 消息收集
    # ------------------------------------------------------------------
    def add_message(self, sender_name: str, sender_id: str, content: str) -> int:
        """
        添加一条群聊消息到缓冲区。
        返回当前缓冲区中的消息数量。
        """
        messages = self._load_messages()
        messages.append({
            "sender_name": sender_name,
            "sender_id": sender_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_messages(messages)

        # 记录参与者
        novel = self._load_novel()
        contributors = novel.get("contributors", [])
        if sender_name and sender_name not in contributors:
            contributors.append(sender_name)
            novel["contributors"] = contributors
            self._save_novel(novel)

        return len(messages)

    def get_pending_count(self) -> int:
        """获取待处理的消息数量"""
        return len(self._load_messages())

    def get_pending_messages(self) -> list:
        """获取所有待处理的消息"""
        return self._load_messages()

    # ------------------------------------------------------------------
    # 人物管理
    # ------------------------------------------------------------------
    def get_character(self, name: str) -> Optional[dict]:
        """通过真名或小说名查找角色"""
        novel = self._load_novel()
        for ch in novel.get("characters", []):
            if ch.get("real_name") == name or ch.get("novel_name") == name:
                return ch
        return None

    def list_characters(self) -> list:
        novel = self._load_novel()
        return novel.get("characters", [])

    def _update_characters(self, new_chars: list[dict]) -> None:
        """更新人物列表（去重合并）"""
        novel = self._load_novel()
        existing = novel.get("characters", [])
        existing_ids = {c.get("sender_id") for c in existing}
        existing_names = {c.get("real_name") for c in existing}

        for ch in new_chars:
            sid = ch.get("sender_id", "")
            rname = ch.get("real_name", "")
            if sid and sid in existing_ids:
                # 更新已有角色的描述
                for e in existing:
                    if e.get("sender_id") == sid:
                        if ch.get("description"):
                            e["description"] = ch["description"]
                        if ch.get("novel_name"):
                            e["novel_name"] = ch["novel_name"]
                        break
            elif rname and rname in existing_names:
                # 按真名去重
                for e in existing:
                    if e.get("real_name") == rname:
                        if ch.get("description"):
                            e["description"] = ch["description"]
                        break
            else:
                existing.append(ch)
                existing_ids.add(sid)
                existing_names.add(rname)

        novel["characters"] = existing
        self._save_novel(novel)

    # ------------------------------------------------------------------
    # 章节管理
    # ------------------------------------------------------------------
    def get_chapters(self) -> list:
        novel = self._load_novel()
        return novel.get("chapters", [])

    def get_chapter_count(self) -> int:
        return len(self.get_chapters())

    def get_chapter_by_number(self, number: int) -> Optional[dict]:
        for ch in self.get_chapters():
            if ch.get("number") == number:
                return ch
        return None

    # ------------------------------------------------------------------
    # AI 消息质量评估
    # ------------------------------------------------------------------
    async def evaluate_quality(self, provider) -> tuple[bool, str]:
        """
        评估当前缓冲区的消息质量是否足以生成章节。
        返回 (sufficient: bool, reason: str)。
        """
        messages = self._load_messages()
        if not messages:
            return False, "没有待处理的消息"

        # 格式化聊天记录
        chat_log_lines = []
        for msg in messages:
            name = msg.get("sender_name", "未知")
            content = msg.get("content", "")
            chat_log_lines.append(f"[{name}]: {content}")
        chat_log = "\n".join(chat_log_lines)

        prompt = CHAT_NOVEL_EVALUATE_QUALITY_PROMPT.format(
            message_count=len(messages),
            chat_log=chat_log[:4000],
        )

        try:
            response = await call_llm(provider, prompt, timeout=30)
            result = parse_json_from_response(response)
            if result:
                sufficient = bool(result.get("sufficient", False))
                reason = result.get("reason", "")
                ratio = result.get("valid_ratio", "")
                logger.info(
                    f"[{PLUGIN_ID}] 群聊小说质量评估："
                    f"sufficient={sufficient}, ratio={ratio}, reason={reason}"
                )
                return sufficient, f"{ratio} — {reason}" if ratio else reason
            # 解析失败，默认放行
            return True, "评估解析失败，默认允许生成"
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 群聊小说质量评估异常: {e}")
            # 异常时默认放行，避免永远不生成
            return True, f"评估异常: {e}"

    # ------------------------------------------------------------------
    # AI 章节生成
    # ------------------------------------------------------------------
    async def generate_chapter(self, provider) -> Optional[dict]:
        """
        从当前消息缓冲生成新的一章。
        返回生成的章节 dict，失败返回 None。
        """
        messages = self._load_messages()
        if not messages:
            return None

        novel = self._load_novel()

        # 格式化聊天记录
        chat_log_lines = []
        participants = set()
        for msg in messages:
            name = msg.get("sender_name", "未知")
            content = msg.get("content", "")
            participants.add(name)
            chat_log_lines.append(f"[{name}]: {content}")
        chat_log = "\n".join(chat_log_lines)

        # 获取已有人物信息
        chars = novel.get("characters", [])
        chars_info = ""
        if chars:
            chars_info = "\n".join([
                f"- {c.get('real_name', '?')} → 小说名: {c.get('novel_name', '?')}，设定: {c.get('description', '暂无')}"
                for c in chars
            ])
        else:
            chars_info = "暂无已有角色，请根据群聊参与者创建角色"

        # 获取前序章节摘要
        previous_chapters = ""
        for ch in novel.get("chapters", []):
            previous_chapters += f"第{ch['number']}章「{ch['title']}」：{ch.get('summary', '无摘要')}\n"
        if not previous_chapters:
            previous_chapters = "这是第一章，没有前序章节。"

        # 新参与者列表（还未映射为角色的）
        existing_names = {c.get("real_name") for c in chars}
        new_participants = [p for p in participants if p not in existing_names]
        new_participants_text = "、".join(new_participants) if new_participants else "无新参与者"

        chapter_number = len(novel.get("chapters", [])) + 1

        # 1. 先映射新参与者为角色（如果有）
        if new_participants:
            try:
                await self._map_new_characters(
                    provider, list(new_participants), novel.get("requirements", "")
                )
                # 重新加载更新后的人物
                novel = self._load_novel()
                chars = novel.get("characters", [])
                chars_info = "\n".join([
                    f"- {c.get('real_name', '?')} → 小说名: {c.get('novel_name', '?')}，设定: {c.get('description', '暂无')}"
                    for c in chars
                ])
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] 群聊小说角色映射失败: {e}")

        # 2. 生成章节
        prompt = CHAT_NOVEL_GENERATE_CHAPTER_PROMPT.format(
            novel_title=novel.get("title", "群聊物语"),
            chapter_number=chapter_number,
            requirements=novel.get("requirements", "无特殊要求"),
            global_summary=novel.get("global_summary", "故事尚未开始"),
            previous_chapters=previous_chapters,
            characters_info=chars_info,
            chat_log=chat_log[:6000],
            new_participants=new_participants_text,
        )

        try:
            response = await call_llm(provider, prompt, timeout=240)
            result = parse_json_from_response(response)

            if not result:
                # 如果 AI 没有返回 JSON，尝试把整个响应当作章节内容
                chapter = {
                    "number": chapter_number,
                    "title": f"第{chapter_number}章",
                    "content": response.strip(),
                    "summary": response.strip()[:200],
                }
            else:
                chapter = {
                    "number": chapter_number,
                    "title": result.get("chapter_title", f"第{chapter_number}章"),
                    "content": result.get("content", response.strip()),
                    "summary": result.get("summary", ""),
                }

            # 保存章节
            novel["chapters"].append(chapter)

            # 更新全局摘要
            if result and result.get("updated_summary"):
                novel["global_summary"] = result["updated_summary"]
            elif chapter.get("summary"):
                novel["global_summary"] = (
                    novel.get("global_summary", "") + " " + chapter["summary"]
                )[-500:]

            # 更新角色信息（如果 AI 返回了额外的角色信息）
            if result and result.get("character_updates"):
                for cu in result["character_updates"]:
                    rname = cu.get("real_name", "")
                    for c in novel.get("characters", []):
                        if c.get("real_name") == rname or c.get("novel_name") == cu.get("novel_name", ""):
                            if cu.get("description"):
                                c["description"] = cu["description"]
                            break

            self._save_novel(novel)

            # 清空消息缓冲
            self._save_messages([])

            logger.info(
                f"[{PLUGIN_ID}] 群聊小说第{chapter_number}章生成完成："
                f"{chapter['title']}（{len(chapter.get('content', ''))}字）"
            )
            return chapter

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 群聊小说章节生成失败: {e}")
            return None

    async def _map_new_characters(
        self, provider, new_names: list[str], requirements: str
    ) -> None:
        """将新的群聊参与者映射为小说角色"""
        novel = self._load_novel()
        existing_chars = novel.get("characters", [])
        existing_info = "\n".join([
            f"- {c.get('real_name')} → {c.get('novel_name')}：{c.get('description', '')}"
            for c in existing_chars
        ]) if existing_chars else "暂无已有角色"

        prompt = CHAT_NOVEL_MAP_CHARACTERS_PROMPT.format(
            new_participants=", ".join(new_names),
            existing_characters=existing_info,
            requirements=requirements or "无特殊要求",
        )

        response = await call_llm(provider, prompt, timeout=60)
        result = parse_json_from_response(response)
        if not result:
            return

        new_chars = result.get("characters", [])
        mapped = []
        for ch in new_chars:
            real_name = ch.get("real_name", "").strip()
            if not real_name or real_name not in new_names:
                continue
            mapped.append({
                "real_name": real_name,
                "novel_name": ch.get("novel_name", real_name).strip(),
                "description": ch.get("description", "").strip(),
                "sender_id": "",
            })

        if mapped:
            self._update_characters(mapped)
            logger.info(f"[{PLUGIN_ID}] 群聊小说角色映射完成：{[c['real_name'] for c in mapped]}")

    # ------------------------------------------------------------------
    # 数据管理
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_chapter_prefix(title: str) -> str:
        """去除标题中已有的 '第N章' 前缀，避免与手动拼接的章节号重复"""
        return _re.sub(r'^第\s*\d+\s*章[：:\s]*', '', title).strip()

    def reset(self) -> None:
        """删除当前群聊的所有小说数据（人物、章节、消息等）"""
        self._save_novel(dict(_DEFAULT_CHAT_NOVEL))
        self._save_messages([])
        logger.info(f"[{PLUGIN_ID}] 群聊小说数据已重置")

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def get_novel_data(self) -> dict:
        """获取小说数据（用于导出）"""
        novel = self._load_novel()
        # 构建简介：风格要求 + 剧情简介（分开展示）
        synopsis_parts = []
        if novel.get("requirements"):
            synopsis_parts.append(f"风格：{novel['requirements']}")
        if novel.get("global_summary") and novel["global_summary"] != "故事尚未开始。":
            synopsis_parts.append(f"\n剧情简介：{novel['global_summary']}")
        synopsis = "\n".join(synopsis_parts)
        # 转换为与现有导出函数兼容的格式
        export_data = {
            "title": novel.get("title", "群聊物语"),
            "synopsis": synopsis,
            "contributors": novel.get("contributors", []),
            "chapters": [],
        }
        for ch in novel.get("chapters", []):
            # 清理标题中可能重复的 "第N章" 前缀
            clean_title = self._strip_chapter_prefix(ch.get("title", ""))
            export_data["chapters"].append({
                "number": ch.get("number", 0),
                "title": clean_title,
                "scenes": [{
                    "title": "",
                    "content": ch.get("content", ""),
                }],
            })
        return export_data

    def export_text(self) -> str:
        """导出全文文本"""
        novel = self._load_novel()
        lines = [f"《{novel.get('title', '群聊物语')}》", ""]
        if novel.get("requirements"):
            lines.append(f"【主题】{novel['requirements']}")
            lines.append("")
        for ch in novel.get("chapters", []):
            clean_title = self._strip_chapter_prefix(ch.get('title', ''))
            lines.append(f"第{ch.get('number', '?')}章 {clean_title}")
            lines.append("=" * 40)
            lines.append("")
            lines.append(ch.get("content", ""))
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 状态查看
    # ------------------------------------------------------------------
    def get_status(self) -> str:
        """获取群聊小说当前状态"""
        novel = self._load_novel()
        status_text = "📡 收集中" if novel.get("status") == "collecting" else "⏹ 已停止"
        chapters = novel.get("chapters", [])
        pending = self.get_pending_count()
        total_chars = sum(len(ch.get("content", "")) for ch in chapters)
        char_count = len(novel.get("characters", []))

        lines = [
            f"📖 群聊小说《{novel.get('title', '群聊物语')}》",
            f"  状态：{status_text}",
            f"  主题要求：{truncate_text(novel.get('requirements', '无'), 60)}",
            f"  已生成章节：{len(chapters)}",
            f"  总字数：{total_chars}",
            f"  人物数：{char_count}",
            f"  待处理消息：{pending} 条",
        ]
        if novel.get("global_summary"):
            lines.append(f"  故事进展：{truncate_text(novel['global_summary'], 200)}")
        return "\n".join(lines)
