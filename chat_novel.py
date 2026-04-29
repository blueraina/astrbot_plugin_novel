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
    CHAT_NOVEL_GENERATE_CHAPTER_TEXT_PROMPT,
    CHAT_NOVEL_EXTRACT_CHAPTER_METADATA_PROMPT,
    CHAT_NOVEL_MAP_CHARACTERS_PROMPT,
    CHAT_NOVEL_EVALUATE_QUALITY_PROMPT,
    CHAT_NOVEL_FILTER_MESSAGES_PROMPT,
    CHAT_NOVEL_RELATIONSHIP_PROMPT,
    CHAT_NOVEL_REWRITE_CHAPTER_PROMPT,
    CHAT_NOVEL_PLOT_CHECK_PROMPT,
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
    "cover_auto_generate": True,  # 导出时是否每次重新生成封面
    "preview_enabled": True,      # 生成章节后是否发送预览文本到群聊
    "custom_settings": [],        # 用户自定义设定列表
    "force_ending": False,        # 下一次生成是否强制结局
    "next_plot_direction": "",    # 下一次生成章节的临时剧情走向
    "story_bible": {},            # HCA 式高压缩故事档案
    "memory_entries": [],         # CSA 式可召回历史记忆块
}


_MAX_MEMORY_ENTRIES = 300


def _fresh_default_chat_novel() -> dict:
    """返回不共享可变对象的默认群聊小说数据。"""
    return _json.loads(_json.dumps(_DEFAULT_CHAT_NOVEL, ensure_ascii=False))


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
        novel = safe_json_load(self._novel_path, _fresh_default_chat_novel())
        if not isinstance(novel, dict):
            return _fresh_default_chat_novel()
        defaults = _fresh_default_chat_novel()
        for key, value in defaults.items():
            if key not in novel:
                novel[key] = value
        return novel

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
    def add_message(
        self, sender_name: str, sender_id: str, content: str,
        image_descriptions: list[str] | None = None,
    ) -> int:
        """
        添加一条群聊消息到缓冲区。
        image_descriptions: 该消息中包含的图片识别结果列表（可选）
        返回当前缓冲区中的消息数量。
        """
        messages = self._load_messages()
        if sender_id and sender_id != "bot":
            sender_name = f"{sender_name}(ID:{sender_id})"

        msg_data: dict = {
            "sender_name": sender_name,
            "sender_id": sender_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if image_descriptions:
            msg_data["image_descriptions"] = image_descriptions
        messages.append(msg_data)
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

    @staticmethod
    def _parse_participant_identity(name: str) -> tuple[str, str]:
        """拆分群聊昵称和 AstrBot 附加的 (ID:xxx)。"""
        text = (name or "").strip()
        m = _re.search(r"\(ID:(.*?)\)\s*$", text, _re.IGNORECASE)
        if not m:
            return text, ""
        display = text[:m.start()].strip() or text
        return display, m.group(1).strip()

    @staticmethod
    def _normalize_name(name: str) -> str:
        display, _ = ChatNovelEngine._parse_participant_identity(name)
        return _re.sub(r"\s+", "", display).lower()

    def _update_characters(self, new_chars: list[dict]) -> None:
        """更新人物列表（去重合并，跳过已锁定的角色）"""
        novel = self._load_novel()
        existing = novel.get("characters", [])
        existing_ids = {c.get("sender_id") for c in existing}
        existing_names = {c.get("real_name") for c in existing}
        existing_display = {
            self._normalize_name(c.get("real_name", "")): c
            for c in existing
            if self._normalize_name(c.get("real_name", ""))
        }

        for ch in new_chars:
            sid = ch.get("sender_id", "")
            rname = ch.get("real_name", "")

            if not sid and "(ID:" in rname:
                _, parsed_sid = self._parse_participant_identity(rname)
                if parsed_sid:
                    sid = parsed_sid
                    ch["sender_id"] = sid

            if sid and sid in existing_ids:
                # 更新已有角色的描述（跳过锁定角色）
                for e in existing:
                    if e.get("sender_id") == sid:
                        if e.get("locked"):
                            break  # 锁定角色不允许 AI 修改
                        if ch.get("description"):
                            e["description"] = ch["description"]
                        if ch.get("novel_name"):
                            e["novel_name"] = ch["novel_name"]
                        break
            elif rname and rname in existing_names:
                # 按真名去重（跳过锁定角色）
                for e in existing:
                    if e.get("real_name") == rname:
                        if e.get("locked"):
                            break
                        if ch.get("description"):
                            e["description"] = ch["description"]
                        break
            elif self._normalize_name(rname) in existing_display:
                e = existing_display[self._normalize_name(rname)]
                if not e.get("locked"):
                    if sid and not e.get("sender_id"):
                        e["sender_id"] = sid
                        existing_ids.add(sid)
                    if ch.get("description"):
                        e["description"] = ch["description"]
                    if ch.get("novel_name"):
                        e["novel_name"] = ch["novel_name"]
            else:
                existing.append(ch)
                existing_ids.add(sid)
                existing_names.add(rname)
                norm = self._normalize_name(rname)
                if norm:
                    existing_display[norm] = ch

        novel["characters"] = existing
        self._save_novel(novel)

    # ------------------------------------------------------------------
    # 章节管理
    # ------------------------------------------------------------------
    def get_chapters(self) -> list:
        novel = self._load_novel()
        return novel.get("chapters", [])

    def get_cover_auto_generate(self) -> bool:
        """获取是否每次导出自动重新生成封面"""
        novel = self._load_novel()
        return novel.get("cover_auto_generate", True)

    def set_cover_auto_generate(self, val: bool) -> None:
        """设置是否每次导出自动重新生成封面"""
        novel = self._load_novel()
        novel["cover_auto_generate"] = val
        self._save_novel(novel)

    def get_preview_enabled(self) -> bool:
        """获取是否在生成章节后发送预览文本"""
        novel = self._load_novel()
        return novel.get("preview_enabled", True)

    def set_preview_enabled(self, val: bool) -> None:
        """设置是否在生成章节后发送预览文本"""
        novel = self._load_novel()
        novel["preview_enabled"] = val
        self._save_novel(novel)

    def get_chapter_count(self) -> int:
        return len(self.get_chapters())

    def get_chapter_by_number(self, number: int) -> Optional[dict]:
        for ch in self.get_chapters():
            if ch.get("number") == number:
                return ch
        return None

    # ------------------------------------------------------------------
    # HCA/CSA 记忆管理
    # ------------------------------------------------------------------
    @staticmethod
    def _as_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "、".join(ChatNovelEngine._as_text(v) for v in value if ChatNovelEngine._as_text(v))
        if isinstance(value, dict):
            parts = []
            for key, val in value.items():
                text = ChatNovelEngine._as_text(val)
                if text:
                    parts.append(f"{key}: {text}")
            return "；".join(parts)
        return str(value).strip()

    @staticmethod
    def _as_list(value) -> list:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [v for v in value if v not in (None, "")]
        return [value]

    @staticmethod
    def _limit_list(value, limit: int = 12) -> list[str]:
        items = []
        for item in ChatNovelEngine._as_list(value):
            text = ChatNovelEngine._as_text(item)
            if text and text not in items:
                items.append(text[:120])
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _extract_search_terms(text: str, limit: int = 1600) -> set[str]:
        """轻量关键词/中文 ngram 提取，用于无 embedding 的 CSA 召回。"""
        if not text:
            return set()
        text = str(text).lower()
        terms: set[str] = set()
        terms.update(_re.findall(r"[a-z0-9_]{2,}", text))

        for seg in _re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if len(seg) <= 8:
                terms.add(seg)
            for n in (2, 3, 4):
                if len(seg) < n:
                    continue
                for i in range(len(seg) - n + 1):
                    terms.add(seg[i:i + n])
                    if len(terms) >= limit:
                        return terms
        return terms

    def _ensure_story_bible(self, novel: dict) -> dict:
        defaults = {
            "mainline": "",
            "current_arc": "",
            "core_conflict": "",
            "narrative_tone": "",
            "world_state": "",
            "character_states": [],
            "unresolved_hooks": [],
            "resolved_hooks": [],
            "important_facts": [],
            "next_direction": "",
            "last_updated_chapter": 0,
        }
        raw = novel.get("story_bible")
        bible = raw if isinstance(raw, dict) else {}
        for key, value in defaults.items():
            if key not in bible:
                bible[key] = value
        if not bible.get("mainline") and novel.get("requirements"):
            bible["mainline"] = f"围绕「{novel.get('requirements', '')}」展开群像故事。"
        if not bible.get("current_arc") and novel.get("global_summary"):
            bible["current_arc"] = novel.get("global_summary", "")
        return bible

    def _format_story_bible(self, novel: dict) -> str:
        bible = self._ensure_story_bible(novel)
        lines = [
            f"主线目标：{bible.get('mainline') or '尚未明确，请在本章建立清晰主线'}",
            f"当前篇章：{bible.get('current_arc') or '尚未明确'}",
            f"核心冲突：{bible.get('core_conflict') or '尚未明确'}",
            f"叙事基调：{bible.get('narrative_tone') or novel.get('requirements') or '遵循用户要求'}",
            f"世界/局势：{bible.get('world_state') or '暂无明确变化'}",
        ]

        def append_list(title: str, items: list[str], empty: str) -> None:
            lines.append(f"{title}：")
            if items:
                for item in items[:12]:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- {empty}")

        append_list("重要角色状态", self._limit_list(bible.get("character_states"), 12), "暂无")
        append_list("未解决伏笔", self._limit_list(bible.get("unresolved_hooks"), 12), "暂无")
        append_list("必须记住的事实", self._limit_list(bible.get("important_facts"), 12), "暂无")
        lines.append(f"下一步方向：{bible.get('next_direction') or '承接上一章自然推进'}")
        return "\n".join(lines)

    def _format_custom_settings(self, novel: dict) -> str:
        settings = novel.get("custom_settings", [])
        if not settings:
            return "无。"
        lines = []
        for i, setting in enumerate(settings, 1):
            lines.append(f"{i}. {setting.get('content', '')}")
        return "\n".join(lines)

    def _format_recent_context(self, novel: dict) -> str:
        chapters = novel.get("chapters", [])
        if not chapters:
            return "这是第一章，没有近邻章节。"
        lines = []
        for ch in chapters[-2:]:
            content = ch.get("content", "")
            ending = content[-500:] if content else ""
            lines.append(
                f"第{ch.get('number')}章「{ch.get('title', '')}」\n"
                f"摘要：{ch.get('summary', '无摘要')}\n"
                f"结尾片段：{ending or '无'}"
            )
        return "\n\n".join(lines)

    def _format_previous_chapters(self, novel: dict, memory_enabled: bool = True) -> str:
        chapters = novel.get("chapters", [])
        if not chapters:
            return "这是第一章，没有前序章节。"
        if not memory_enabled or len(chapters) <= 6:
            return "\n".join(
                f"第{ch.get('number')}章「{ch.get('title', '')}」：{ch.get('summary', '无摘要')}"
                for ch in chapters
            )
        latest = chapters[-3:]
        lines = [
            f"已有 {len(chapters)} 章。较早章节细节请优先参考 HCA 故事档案和 CSA 相关记忆召回。",
            "最近章节概要：",
        ]
        lines.extend(
            f"第{ch.get('number')}章「{ch.get('title', '')}」：{ch.get('summary', '无摘要')}"
            for ch in latest
        )
        return "\n".join(lines)

    def _retrieve_relevant_memories(
        self,
        novel: dict,
        chat_log: str,
        participants: set[str],
        chars: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        if top_k <= 0:
            return []

        bible = self._ensure_story_bible(novel)
        query_parts = [
            chat_log[:4000],
            novel.get("requirements", ""),
            bible.get("mainline", ""),
            bible.get("current_arc", ""),
            bible.get("core_conflict", ""),
            bible.get("next_direction", ""),
            " ".join(participants),
        ]
        query_text = "\n".join(query_parts)
        query_terms = self._extract_search_terms(query_text)

        active_names: set[str] = set()
        for name in participants:
            name_text = str(name).lower()
            if name_text:
                active_names.add(name_text)
            m = _re.search(r"^(.*?)\(id:.*?\)$", name_text)
            if m and m.group(1).strip():
                active_names.add(m.group(1).strip())
        for ch in chars:
            real = ch.get("real_name", "")
            novel_name = ch.get("novel_name", "")
            sender_id = str(ch.get("sender_id", ""))
            if (
                (real and real in query_text)
                or (novel_name and novel_name in query_text)
                or (sender_id and sender_id in query_text)
            ):
                active_names.add(real.lower())
                active_names.add(novel_name.lower())
                query_terms.update(self._extract_search_terms(f"{real} {novel_name}", 100))

        chapters = novel.get("chapters", [])
        latest_number = max([int(ch.get("number", 0) or 0) for ch in chapters] + [1])

        candidates: list[dict] = []
        for entry in novel.get("memory_entries", []):
            if isinstance(entry, dict):
                candidates.append(dict(entry))

        known_chapter_entries = {
            int(e.get("chapter_number", 0) or 0)
            for e in candidates
            if isinstance(e, dict) and e.get("type") == "chapter_summary"
        }
        for ch in chapters:
            number = int(ch.get("number", 0) or 0)
            if number in known_chapter_entries:
                continue
            candidates.append({
                "id": f"chapter-{number}-summary",
                "type": "chapter_summary",
                "chapter_number": number,
                "chapter_title": ch.get("title", ""),
                "title": f"第{number}章概要",
                "content": ch.get("summary") or truncate_text(ch.get("content", ""), 220),
                "characters": [],
                "keywords": [],
                "importance": 3,
            })

        scored: list[tuple[float, dict]] = []
        for entry in candidates:
            content = " ".join([
                self._as_text(entry.get("title")),
                self._as_text(entry.get("content")),
                self._as_text(entry.get("chapter_title")),
                self._as_text(entry.get("characters")),
                self._as_text(entry.get("keywords")),
            ])
            entry_terms = self._extract_search_terms(content)
            if not entry_terms:
                continue
            overlap = len(query_terms & entry_terms)
            chars_text = self._as_text(entry.get("characters")).lower()
            char_overlap = sum(1 for name in active_names if name and name in chars_text)
            try:
                importance = int(entry.get("importance", 3) or 3)
            except Exception:
                importance = 3
            chapter_number = int(entry.get("chapter_number", 0) or 0)
            recency = chapter_number / latest_number if latest_number else 0
            entry_type = self._as_text(entry.get("type")).lower()
            type_boost = 0.0
            if any(key in entry_type for key in ("hook", "伏笔", "conflict", "冲突", "mainline", "主线")):
                type_boost += 2.0
            if chapter_number and latest_number - chapter_number <= 2:
                type_boost += 1.2
            score = overlap * 2.4 + char_overlap * 8.0 + importance * 1.2 + recency * 2.0 + type_boost
            if score > 0:
                entry["_score"] = round(score, 2)
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _format_relevant_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "暂无可召回的相关历史记忆。"
        lines = []
        for i, entry in enumerate(memories, 1):
            chapter = entry.get("chapter_number")
            chapter_part = f"第{chapter}章" if chapter else "历史记忆"
            title = entry.get("title") or entry.get("chapter_title") or "未命名记忆"
            content = self._as_text(entry.get("content")) or "无内容"
            chars = self._as_text(entry.get("characters"))
            keywords = self._as_text(entry.get("keywords"))
            tail = []
            if chars:
                tail.append(f"角色：{chars}")
            if keywords:
                tail.append(f"关键词：{keywords}")
            suffix = f"（{'；'.join(tail)}）" if tail else ""
            lines.append(f"{i}. {chapter_part} [{entry.get('type', 'memory')}] {title}：{content}{suffix}")
        return "\n".join(lines)

    def _merge_story_bible_from_result(self, novel: dict, chapter: dict, result: Optional[dict]) -> None:
        bible = self._ensure_story_bible(novel)
        raw = result.get("story_bible") if isinstance(result, dict) else None
        if isinstance(raw, dict):
            for key in (
                "mainline", "current_arc", "core_conflict", "narrative_tone",
                "world_state", "next_direction",
            ):
                text = self._as_text(raw.get(key))
                if text:
                    bible[key] = text[:600]
            for key in ("character_states", "unresolved_hooks", "resolved_hooks", "important_facts"):
                merged = self._limit_list(raw.get(key), 16)
                if merged:
                    bible[key] = merged

        if not bible.get("current_arc") and chapter.get("summary"):
            bible["current_arc"] = chapter["summary"]
        if not bible.get("mainline") and novel.get("requirements"):
            bible["mainline"] = f"围绕「{novel.get('requirements', '')}」展开群像故事。"
        if not bible.get("next_direction") and chapter.get("summary"):
            bible["next_direction"] = f"承接第{chapter.get('number')}章：{chapter.get('summary')}"
        bible["last_updated_chapter"] = chapter.get("number", 0)
        novel["story_bible"] = bible

    def _add_chapter_memory_entries(self, novel: dict, chapter: dict, result: Optional[dict]) -> None:
        chapter_number = int(chapter.get("number", 0) or 0)
        entries = [
            e for e in novel.get("memory_entries", [])
            if not (isinstance(e, dict) and int(e.get("chapter_number", 0) or 0) == chapter_number)
        ]
        raw_entries = result.get("memory_entries") if isinstance(result, dict) else None
        new_entries: list[dict] = []
        if isinstance(raw_entries, list):
            for idx, raw in enumerate(raw_entries[:12], 1):
                if not isinstance(raw, dict):
                    continue
                content = self._as_text(raw.get("content"))
                if not content:
                    continue
                new_entries.append({
                    "id": f"ch{chapter_number}-{idx}-{generate_id()[:6]}",
                    "type": self._as_text(raw.get("type")) or "event",
                    "chapter_number": chapter_number,
                    "chapter_title": chapter.get("title", ""),
                    "title": self._as_text(raw.get("title")) or f"第{chapter_number}章记忆",
                    "content": content[:500],
                    "characters": self._limit_list(raw.get("characters"), 8),
                    "keywords": self._limit_list(raw.get("keywords"), 10),
                    "importance": raw.get("importance", 3),
                    "created_at": datetime.now().isoformat(),
                })

        if not new_entries:
            new_entries.append({
                "id": f"ch{chapter_number}-summary-{generate_id()[:6]}",
                "type": "chapter_summary",
                "chapter_number": chapter_number,
                "chapter_title": chapter.get("title", ""),
                "title": f"第{chapter_number}章概要",
                "content": chapter.get("summary") or truncate_text(chapter.get("content", ""), 240),
                "characters": [],
                "keywords": self._limit_list([
                    chapter.get("title", ""),
                    novel.get("title", ""),
                ], 6),
                "importance": 3,
                "created_at": datetime.now().isoformat(),
            })

        entries.extend(new_entries)
        novel["memory_entries"] = entries[-_MAX_MEMORY_ENTRIES:]

    @staticmethod
    def _strip_wrapping_code_fence(text: str) -> str:
        text = (text or "").strip()
        m = _re.match(r"^```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```\s*$", text, _re.DOTALL)
        if m:
            return m.group(1).strip()
        return text

    @staticmethod
    def _extract_json_string_field(text: str, field: str) -> str:
        """从完整或残缺 JSON 文本中尽量提取字符串字段。"""
        if not text:
            return ""
        m = _re.search(rf'"{_re.escape(field)}"\s*:\s*"', text, _re.DOTALL)
        if not m:
            return ""

        i = m.end()
        chars: list[str] = []
        escaping = False
        while i < len(text):
            ch = text[i]
            if escaping:
                if ch == "n":
                    chars.append("\n")
                elif ch == "r":
                    chars.append("\r")
                elif ch == "t":
                    chars.append("\t")
                elif ch == "b":
                    chars.append("\b")
                elif ch == "f":
                    chars.append("\f")
                elif ch == "u" and i + 4 < len(text):
                    code = text[i + 1:i + 5]
                    try:
                        chars.append(chr(int(code, 16)))
                        i += 4
                    except ValueError:
                        chars.append(ch)
                else:
                    chars.append(ch)
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                break
            else:
                chars.append(ch)
            i += 1
        return "".join(chars).strip()

    def _clean_chapter_content(self, text: str) -> str:
        text = self._strip_wrapping_code_fence(text)
        probe = text.lstrip()
        if probe.startswith("{") or probe.startswith("```json") or '"content"' in probe[:500]:
            extracted = self._extract_json_string_field(probe, "content")
            if extracted:
                return self._strip_wrapping_code_fence(extracted).strip()
        return text.strip()

    @staticmethod
    def _strip_leading_chapter_heading(content: str, title: str = "") -> str:
        """移除模型误写进正文开头的章节标题行。"""
        if not content:
            return ""
        lines = content.splitlines()
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            return content.strip()

        first = lines[idx].strip().lstrip("#").strip()
        normalized_first = _re.sub(r"[\s　：:，,。、《》「」\"'“”‘’\-—_]+", "", first)
        normalized_title = _re.sub(r"[\s　：:，,。、《》「」\"'“”‘’\-—_]+", "", title or "")
        is_chapter_heading = bool(
            _re.match(r"^第\s*[\d一二三四五六七八九十百千万零〇两]+\s*章", first)
        )
        is_duplicate_title = bool(
            normalized_title and normalized_first == normalized_title
        )

        if len(first) <= 80 and (is_chapter_heading or is_duplicate_title):
            del lines[idx]
            while idx < len(lines) and not lines[idx].strip():
                del lines[idx]
            return "\n".join(lines).strip()
        return content.strip()

    async def _extract_chapter_metadata(
        self,
        provider,
        novel: dict,
        chapter_number: int,
        chapter_content: str,
        story_bible: str,
        recent_context: str,
        relevant_memories: str,
        chars_info: str,
        chat_log: str,
        next_plot_direction: str = "",
    ) -> dict:
        prompt = CHAT_NOVEL_EXTRACT_CHAPTER_METADATA_PROMPT.format(
            novel_title=novel.get("title", "群聊物语"),
            chapter_number=chapter_number,
            requirements=novel.get("requirements", "无特殊要求"),
            next_plot_direction=next_plot_direction or "无。",
            story_bible=story_bible[:3000],
            global_summary=novel.get("global_summary", "故事尚未开始"),
            recent_context=recent_context[:1800],
            relevant_memories=relevant_memories[:2500],
            characters_info=chars_info[:2500],
            chat_log=chat_log[:3000],
            chapter_content=chapter_content[:6000],
        )
        try:
            response = await call_llm(provider, prompt, timeout=120)
            result = parse_json_from_response(response)
            if isinstance(result, dict):
                return result
            logger.warning(f"[{PLUGIN_ID}] 群聊小说元数据提取结果解析失败，将使用兜底摘要")
            return {}
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 群聊小说元数据提取失败: {e}")
            return {}

    async def _run_plot_check(
        self,
        provider,
        chapter: dict,
        story_bible: str,
        recent_context: str,
        relevant_memories: str,
        chat_log: str,
    ) -> Optional[dict]:
        prompt = CHAT_NOVEL_PLOT_CHECK_PROMPT.format(
            story_bible=story_bible[:3000],
            recent_context=recent_context[:1800],
            relevant_memories=relevant_memories[:2500],
            chat_log=chat_log[:2500],
            chapter_title=chapter.get("title", ""),
            chapter_content=chapter.get("content", "")[:5000],
        )
        try:
            response = await call_llm(provider, prompt, timeout=120)
            result = parse_json_from_response(response)
            if not isinstance(result, dict):
                return None
            return {
                "passed": bool(result.get("passed", True)),
                "mainline_progress_score": result.get("mainline_progress_score", 0),
                "continuity_score": result.get("continuity_score", 0),
                "conflict_score": result.get("conflict_score", 0),
                "hook_score": result.get("hook_score", 0),
                "issues": self._limit_list(result.get("issues"), 8),
                "suggestions": self._limit_list(result.get("suggestions"), 8),
                "summary": self._as_text(result.get("summary"))[:300],
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 群聊小说剧情检查失败: {e}")
            return None

    # ------------------------------------------------------------------
    # AI 消息质量评估
    # ------------------------------------------------------------------
    async def evaluate_quality(
        self, provider, timeout: int = 30, quality_threshold: int = 40
    ) -> tuple[bool, str]:
        """
        评估当前缓冲区的消息质量是否足以生成章节。
        quality_threshold: 有效消息占比阈值（0-100），低于此值判定为不足。
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
            line = f"[{name}]: {content}"
            # 附加图片识别结果
            img_descs = msg.get("image_descriptions", [])
            if img_descs:
                for desc in img_descs:
                    line += f"\n  [图片内容: {desc}]"
            chat_log_lines.append(line)
        chat_log = "\n".join(chat_log_lines)

        prompt = CHAT_NOVEL_EVALUATE_QUALITY_PROMPT.format(
            message_count=len(messages),
            chat_log=chat_log[:4000],
            quality_threshold=quality_threshold,
        )

        try:
            response = await call_llm(provider, prompt, timeout=timeout)
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
    # AI 消息过滤（小模型）
    # ------------------------------------------------------------------
    async def filter_messages(
        self, provider, timeout: int = 60
    ) -> tuple[int, int]:
        """
        使用小模型过滤无用消息，只保留有效消息。
        返回 (原始消息数, 保留消息数)。
        """
        messages = self._load_messages()
        if not messages:
            return 0, 0

        original_count = len(messages)

        # 格式化带序号的聊天记录
        chat_log_lines = []
        for i, msg in enumerate(messages):
            name = msg.get("sender_name", "未知")
            content = msg.get("content", "")
            line = f"[{i}] [{name}]: {content}"
            img_descs = msg.get("image_descriptions", [])
            if img_descs:
                for desc in img_descs:
                    line += f"\n  [图片内容: {desc}]"
            chat_log_lines.append(line)
        chat_log = "\n".join(chat_log_lines)

        prompt = CHAT_NOVEL_FILTER_MESSAGES_PROMPT.format(
            message_count=original_count,
            chat_log=chat_log[:6000],
        )

        try:
            response = await call_llm(provider, prompt, timeout=timeout)
            result = parse_json_from_response(response)
            if result and "keep_indices" in result:
                keep_indices = set(result["keep_indices"])
                filtered = [
                    msg for i, msg in enumerate(messages)
                    if i in keep_indices
                ]
                if filtered:  # 防止全部被过滤
                    self._save_messages(filtered)
                    kept = len(filtered)
                    logger.info(
                        f"[{PLUGIN_ID}] 群聊小说消息过滤完成："
                        f"{original_count} → {kept}"
                    )
                    return original_count, kept
                else:
                    logger.warning(
                        f"[{PLUGIN_ID}] 过滤后无消息保留，保持原样"
                    )
                    return original_count, original_count
            else:
                logger.warning(
                    f"[{PLUGIN_ID}] 消息过滤结果解析失败，保持原样"
                )
                return original_count, original_count
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 群聊小说消息过滤异常: {e}")
            return original_count, original_count

    # ------------------------------------------------------------------
    # AI 章节生成
    # ------------------------------------------------------------------
    async def generate_chapter(
        self, provider, max_word_count: int = 2000, force_ending: bool = False,
        memory_enabled: bool = True, memory_top_k: int = 8,
        plot_check_enabled: bool = True,
    ) -> Optional[dict]:
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
            line = f"[{name}]: {content}"
            # 附加图片识别结果
            img_descs = msg.get("image_descriptions", [])
            if img_descs:
                for desc in img_descs:
                    line += f"\n  [图片内容: {desc}]"
            chat_log_lines.append(line)
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

        # 新参与者列表（还未映射为角色的）
        existing_names = {c.get("real_name") for c in chars}
        existing_ids = {str(c.get("sender_id")) for c in chars if c.get("sender_id")}
        existing_display_names = {
            self._normalize_name(c.get("real_name", ""))
            for c in chars
            if self._normalize_name(c.get("real_name", ""))
        }

        new_participants = []
        for p in participants:
            if p in existing_names:
                continue
            display_name, sender_id = self._parse_participant_identity(p)
            if sender_id and str(sender_id) in existing_ids:
                continue
            if self._normalize_name(display_name) in existing_display_names:
                continue
            new_participants.append(p)

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
                ]) if chars else "暂无已有角色，请根据群聊参与者创建角色"
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] 群聊小说角色映射失败: {e}")

        # 2. 生成章节
        previous_chapters = self._format_previous_chapters(
            novel, memory_enabled=memory_enabled
        )
        story_bible_text = (
            self._format_story_bible(novel)
            if memory_enabled else "未启用 HCA 故事档案。"
        )
        recent_context = self._format_recent_context(novel)
        retrieved_memories = (
            self._retrieve_relevant_memories(
                novel, chat_log, participants, chars, top_k=memory_top_k
            )
            if memory_enabled else []
        )
        relevant_memories_text = self._format_relevant_memories(retrieved_memories)
        custom_settings_text = self._format_custom_settings(novel)
        next_plot_direction = (novel.get("next_plot_direction") or "").strip()
        next_plot_direction_text = next_plot_direction or "无。"

        if memory_enabled:
            logger.info(
                f"[{PLUGIN_ID}] 群聊小说 HCA/CSA 上下文："
                f"召回 {len(retrieved_memories)} 条相关记忆"
            )

        # 构建结局指令（如有）
        ending_instruction = ""
        if force_ending:
            ending_instruction = (
                "\n## ⚠️ 重要：强制结局要求\n"
                "这是本小说的**最后一章**，你必须在本章中为整个故事写出一个完整的结局。\n"
                "要求：总结所有主要角色的命运、解决悬而未决的情节线、给出明确的故事结尾。\n"
                "结尾要有仪式感，让读者感受到故事的圆满收束。\n\n"
            )

        prompt = CHAT_NOVEL_GENERATE_CHAPTER_TEXT_PROMPT.format(
            novel_title=novel.get("title", "群聊物语"),
            chapter_number=chapter_number,
            requirements=novel.get("requirements", "无特殊要求"),
            custom_settings=custom_settings_text,
            next_plot_direction=next_plot_direction_text,
            story_bible=story_bible_text,
            global_summary=novel.get("global_summary", "故事尚未开始"),
            recent_context=recent_context,
            previous_chapters=previous_chapters,
            relevant_memories=relevant_memories_text,
            characters_info=chars_info,
            chat_log=chat_log[:6000],
            new_participants=new_participants_text,
            max_word_count=max_word_count,
            ending_instruction=ending_instruction,
        )

        try:
            response = await call_llm(provider, prompt, timeout=240)
            response_probe = response.lstrip()
            legacy_result = None
            if (
                response_probe.startswith("{")
                or response_probe.startswith("```json")
                or '"content"' in response_probe[:800]
            ):
                legacy_result = parse_json_from_response(response)
            if not isinstance(legacy_result, dict) or not legacy_result.get("content"):
                legacy_result = None

            if legacy_result:
                # 兼容旧提示词或模型自作主张返回 JSON 的情况。
                chapter_content = self._clean_chapter_content(legacy_result.get("content", ""))
                result = legacy_result
                logger.info(f"[{PLUGIN_ID}] 群聊小说正文生成返回旧 JSON，已按兼容模式解析")
            else:
                chapter_content = self._clean_chapter_content(response)
                result = await self._extract_chapter_metadata(
                    provider,
                    novel,
                    chapter_number,
                    chapter_content,
                    story_bible_text,
                    recent_context,
                    relevant_memories_text,
                    chars_info,
                    chat_log,
                    next_plot_direction_text,
                )

            if not chapter_content:
                logger.error(f"[{PLUGIN_ID}] 群聊小说章节正文为空")
                return None

            raw_title = result.get("chapter_title", f"第{chapter_number}章") if result else f"第{chapter_number}章"
            clean_title = self._strip_chapter_prefix(raw_title) or "未命名"
            chapter_content = self._strip_leading_chapter_heading(chapter_content, clean_title)
            if not chapter_content:
                logger.error(f"[{PLUGIN_ID}] 群聊小说章节正文清理后为空")
                return None

            chapter = {
                "number": chapter_number,
                "title": clean_title,
                "content": chapter_content,
                "summary": result.get("summary", "") if result else "",
            }
            if not chapter.get("summary"):
                chapter["summary"] = truncate_text(chapter.get("content", ""), 200)

            # 保存章节
            novel["chapters"].append(chapter)

            # 更新全局摘要
            if result and result.get("updated_summary"):
                novel["global_summary"] = result["updated_summary"]
            elif chapter.get("summary"):
                novel["global_summary"] = (
                    novel.get("global_summary", "") + " " + chapter["summary"]
                )[-500:]

            # 更新角色信息（如果 AI 返回了额外的角色信息，跳过锁定角色）
            if result and result.get("character_updates"):
                for cu in result["character_updates"]:
                    rname = cu.get("real_name", "")
                    for c in novel.get("characters", []):
                        if c.get("real_name") == rname or c.get("novel_name") == cu.get("novel_name", ""):
                            if c.get("locked"):
                                break  # 锁定角色不允许 AI 修改
                            if cu.get("description"):
                                c["description"] = cu["description"]
                            break

            if memory_enabled:
                self._merge_story_bible_from_result(novel, chapter, result)
                self._add_chapter_memory_entries(novel, chapter, result)

            if plot_check_enabled:
                plot_check = await self._run_plot_check(
                    provider,
                    chapter,
                    self._format_story_bible(novel),
                    recent_context,
                    relevant_memories_text,
                    chat_log,
                )
                if plot_check:
                    chapter["plot_check"] = plot_check
                    if not plot_check.get("passed", True):
                        logger.warning(
                            f"[{PLUGIN_ID}] 群聊小说剧情检查未通过："
                            f"{plot_check.get('summary', '')}"
                        )

            if next_plot_direction:
                chapter["user_plot_direction"] = next_plot_direction
                self.clear_next_plot_direction(novel)

            self._save_novel(novel)

            # 清空消息缓冲
            self._save_messages([])

            # 强制结局：生成完成后停止收集并重置标记
            if force_ending:
                novel["force_ending"] = False
                novel["status"] = "stopped"
                self._save_novel(novel)
                logger.info(f"[{PLUGIN_ID}] 群聊小说强制结局完成，已停止收集")

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

        try:
            response = await call_llm(provider, prompt, timeout=60)
            result = parse_json_from_response(response)
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 群聊小说角色映射 AI 调用失败，将使用兜底角色: {e}")
            result = None

        new_chars = result.get("characters", []) if isinstance(result, dict) else []
        mapped = []
        participant_map = {}
        for raw_name in new_names:
            display_name, sender_id = self._parse_participant_identity(raw_name)
            participant_map[raw_name] = {
                "raw": raw_name,
                "display": display_name,
                "sender_id": sender_id,
                "norm": self._normalize_name(display_name),
            }
        matched_raw: set[str] = set()

        def find_match(real_name: str) -> Optional[dict]:
            if real_name in participant_map and real_name not in matched_raw:
                return participant_map[real_name]
            display_name, sender_id = self._parse_participant_identity(real_name)
            norm = self._normalize_name(display_name)
            if sender_id:
                for info in participant_map.values():
                    if info["raw"] not in matched_raw and info["sender_id"] == sender_id:
                        return info
            if norm:
                for info in participant_map.values():
                    if info["raw"] not in matched_raw and info["norm"] == norm:
                        return info
            if len(participant_map) - len(matched_raw) == 1:
                for info in participant_map.values():
                    if info["raw"] not in matched_raw:
                        return info
            return None

        for ch in new_chars:
            real_name = ch.get("real_name", "").strip()
            if not real_name:
                continue
            matched = find_match(real_name)
            if not matched:
                continue
            matched_raw.add(matched["raw"])

            mapped.append({
                "real_name": matched["raw"],
                "novel_name": (ch.get("novel_name") or matched["display"] or matched["raw"]).strip(),
                "description": ch.get("description", "").strip(),
                "sender_id": matched["sender_id"],
            })

        # AI 可能省略 ID 或漏掉部分参与者；未匹配到的群友仍创建兜底角色，避免后续出场人物不更新。
        for info in participant_map.values():
            if info["raw"] in matched_raw:
                continue
            mapped.append({
                "real_name": info["raw"],
                "novel_name": info["display"] or info["raw"],
                "description": "暂无描述",
                "sender_id": info["sender_id"],
            })

        if mapped:
            self._update_characters(mapped)
            logger.info(f"[{PLUGIN_ID}] 群聊小说角色映射完成：{[c['real_name'] for c in mapped]}")

    # ------------------------------------------------------------------
    # 用户自定义设定
    # ------------------------------------------------------------------
    def add_custom_setting(self, content: str) -> int:
        """添加用户自定义设定，返回当前设定总数"""
        novel = self._load_novel()
        settings = novel.get("custom_settings", [])
        settings.append({
            "content": content,
            "added_at": datetime.now().isoformat(),
        })
        novel["custom_settings"] = settings
        self._save_novel(novel)
        logger.info(f"[{PLUGIN_ID}] 群聊小说添加自定义设定：{content[:50]}")
        return len(settings)

    def get_custom_settings(self) -> list:
        """获取所有自定义设定"""
        novel = self._load_novel()
        return novel.get("custom_settings", [])

    def set_next_plot_direction(self, content: str) -> None:
        """设置下一次章节生成的临时剧情走向。"""
        novel = self._load_novel()
        novel["next_plot_direction"] = (content or "").strip()
        self._save_novel(novel)

    def get_next_plot_direction(self) -> str:
        """获取下一次章节生成的临时剧情走向。"""
        novel = self._load_novel()
        return (novel.get("next_plot_direction") or "").strip()

    def clear_next_plot_direction(self, novel: Optional[dict] = None) -> None:
        """清空下一次章节生成的临时剧情走向。"""
        if novel is None:
            novel = self._load_novel()
            novel["next_plot_direction"] = ""
            self._save_novel(novel)
        else:
            novel["next_plot_direction"] = ""

    # ------------------------------------------------------------------
    # 强制结局
    # ------------------------------------------------------------------
    def set_force_ending(self, val: bool) -> None:
        """设置是否在下一次生成时强制结局"""
        novel = self._load_novel()
        novel["force_ending"] = val
        self._save_novel(novel)

    def get_force_ending(self) -> bool:
        """获取是否需要强制结局"""
        novel = self._load_novel()
        return novel.get("force_ending", False)

    # ------------------------------------------------------------------
    # 角色编辑 & 锁定
    # ------------------------------------------------------------------
    def update_character_desc(self, name: str, new_desc: str) -> Optional[dict]:
        """通过真名或小说名修改角色描述，返回修改后的角色或 None"""
        novel = self._load_novel()
        for c in novel.get("characters", []):
            if c.get("real_name") == name or c.get("novel_name") == name:
                c["description"] = new_desc
                self._save_novel(novel)
                return c
        return None

    def toggle_character_lock(self, name: str) -> Optional[tuple[dict, bool]]:
        """
        切换角色的锁定状态。
        返回 (角色dict, 新的锁定状态) 或 None（角色不存在）。
        """
        novel = self._load_novel()
        for c in novel.get("characters", []):
            if c.get("real_name") == name or c.get("novel_name") == name:
                new_locked = not c.get("locked", False)
                c["locked"] = new_locked
                self._save_novel(novel)
                return c, new_locked
        return None


    # ------------------------------------------------------------------
    # 角色关系图
    # ------------------------------------------------------------------
    async def generate_relationship_graph(self, provider, timeout: int = 120) -> Optional[dict]:
        """AI 生成角色关系 Mermaid 代码"""
        novel = self._load_novel()
        chars = novel.get("characters", [])
        chapters = novel.get("chapters", [])
        if not chars or not chapters:
            return None

        chars_info = "\n".join([
            f"- {c.get('novel_name', '?')}（原型：{c.get('real_name', '?')}）：{c.get('description', '暂无')}"
            for c in chars
        ])

        chapters_summary = "\n".join([
            f"第{ch['number']}章「{ch.get('title', '')}」：{ch.get('summary', truncate_text(ch.get('content', ''), 200))}"
            for ch in chapters
        ])

        prompt = CHAT_NOVEL_RELATIONSHIP_PROMPT.format(
            characters_info=chars_info,
            chapters_summary=chapters_summary[:4000],
        )

        try:
            response = await call_llm(provider, prompt, timeout=timeout)
            result = parse_json_from_response(response)
            if result and "mermaid_code" in result:
                code = result["mermaid_code"]
                # 移除 AI 可能生成的 %%{init}%% 指令（主题由 URL 参数控制）
                import re as _re_local
                code = _re_local.sub(r'%%\{init:.*?\}%%\s*', '', code)
                # 确保有 linkStyle 让连线清晰可见（粗线 + 深色）
                if "linkStyle" not in code:
                    code += "\n    linkStyle default stroke:#333,stroke-width:3px"
                else:
                    # 替换已有的 linkStyle 让线条更粗
                    code = _re_local.sub(
                        r'linkStyle\s+default\s+stroke:[^,]+,stroke-width:\d+px',
                        'linkStyle default stroke:#333,stroke-width:3px',
                        code,
                    )
                result["mermaid_code"] = code
            return result
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 关系图生成失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 章节重写
    # ------------------------------------------------------------------
    async def rewrite_chapter(
        self, provider, chapter_number: int,
        instructions: str = "", max_word_count: int = 2000
    ) -> Optional[dict]:
        """重写指定章节，返回新章节 dict 或 None"""
        novel = self._load_novel()
        chapters = novel.get("chapters", [])

        # 查找目标章节
        target_idx = None
        target_ch = None
        for i, ch in enumerate(chapters):
            if ch.get("number") == chapter_number:
                target_idx = i
                target_ch = ch
                break
        if target_ch is None:
            return None

        # 角色信息
        chars = novel.get("characters", [])
        chars_info = "\n".join([
            f"- {c.get('real_name', '?')} → {c.get('novel_name', '?')}：{c.get('description', '暂无')}"
            for c in chars
        ]) if chars else "暂无角色"

        # 前序章节
        previous_chapters = ""
        for ch in chapters[:target_idx]:
            previous_chapters += f"第{ch['number']}章「{ch.get('title', '')}」：{ch.get('summary', '无摘要')}\n"
        if not previous_chapters:
            previous_chapters = "这是第一章，没有前序章节。"

        # 后续章节
        next_chapters = ""
        for ch in chapters[target_idx + 1:]:
            next_chapters += f"第{ch['number']}章「{ch.get('title', '')}」：{ch.get('summary', '无摘要')}\n"
        if not next_chapters:
            next_chapters = "这是最新章节，没有后续章节。"

        prompt = CHAT_NOVEL_REWRITE_CHAPTER_PROMPT.format(
            novel_title=novel.get("title", "群聊物语"),
            chapter_number=chapter_number,
            requirements=novel.get("requirements", "无特殊要求"),
            global_summary=novel.get("global_summary", "故事尚未开始"),
            previous_chapters=previous_chapters,
            next_chapters=next_chapters,
            characters_info=chars_info,
            original_content=truncate_text(target_ch.get("content", ""), 4000),
            user_instructions=instructions or "请保持原有风格，优化内容质量",
            max_word_count=max_word_count,
        )

        try:
            response = await call_llm(provider, prompt, timeout=240)
            response_probe = response.lstrip()
            result = None
            if (
                response_probe.startswith("{")
                or response_probe.startswith("```json")
                or '"content"' in response_probe[:800]
            ):
                result = parse_json_from_response(response)
            if not isinstance(result, dict) or not result.get("content"):
                result = None

            if not result:
                content = self._clean_chapter_content(response)
                title = self._strip_chapter_prefix(target_ch.get("title", f"第{chapter_number}章")) or "未命名"
                content = self._strip_leading_chapter_heading(content, title)
                new_chapter = {
                    "number": chapter_number,
                    "title": title,
                    "content": content,
                    "summary": truncate_text(content, 200),
                }
            else:
                content = self._clean_chapter_content(result.get("content", ""))
                raw_title = result.get("chapter_title", target_ch.get("title", f"第{chapter_number}章"))
                title = self._strip_chapter_prefix(raw_title) or "未命名"
                content = self._strip_leading_chapter_heading(content, title)
                new_chapter = {
                    "number": chapter_number,
                    "title": title,
                    "content": content,
                    "summary": result.get("summary", ""),
                }
            if not new_chapter.get("content"):
                logger.error(f"[{PLUGIN_ID}] 群聊小说章节重写正文为空")
                return None
            if not new_chapter.get("summary"):
                new_chapter["summary"] = truncate_text(new_chapter.get("content", ""), 200)

            # 替换原章节
            chapters[target_idx] = new_chapter
            novel["chapters"] = chapters
            self._add_chapter_memory_entries(novel, new_chapter, result)
            self._save_novel(novel)

            logger.info(
                f"[{PLUGIN_ID}] 群聊小说第{chapter_number}章重写完成："
                f"{new_chapter['title']}（{len(new_chapter.get('content', ''))}字）"
            )
            return new_chapter

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 群聊小说章节重写失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 数据管理
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_chapter_prefix(title: str) -> str:
        """去除标题中已有的 '第N章' 前缀，避免与手动拼接的章节号重复"""
        return _re.sub(
            r'^第\s*[\d一二三四五六七八九十百千万零〇两]+\s*章[：:\s、，.\-—]*',
            '',
            title or '',
        ).strip()

    def reset(self) -> None:
        """删除当前群聊的所有小说数据（人物、章节、消息等）"""
        self._save_novel(_fresh_default_chat_novel())
        self._save_messages([])
        logger.info(f"[{PLUGIN_ID}] 群聊小说数据已重置")

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def get_novel_data(self) -> dict:
        """获取小说数据（用于导出）"""
        novel = self._load_novel()
        # 构建简介：风格要求 + 自定义设定 + 剧情简介（分开展示）
        synopsis_parts = []
        if novel.get("requirements"):
            synopsis_parts.append(f"风格：{novel['requirements']}")
        # 添加用户自定义设定
        custom_settings = novel.get("custom_settings", [])
        if custom_settings:
            synopsis_parts.append("")
            synopsis_parts.append("用户设定：")
            for i, s in enumerate(custom_settings, 1):
                synopsis_parts.append(f"  {i}. {s.get('content', '')}")
        if novel.get("global_summary") and novel["global_summary"] != "故事尚未开始。":
            synopsis_parts.append(f"\n剧情简介：{novel['global_summary']}")
        synopsis = "\n".join(synopsis_parts)
        # 转换为与现有导出函数兼容的格式
        export_data = {
            "title": novel.get("title", "群聊物语"),
            "synopsis": synopsis,
            "contributors": novel.get("contributors", []),
            "characters": novel.get("characters", []),
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
                    "content": self._strip_leading_chapter_heading(
                        ch.get("content", ""), clean_title
                    ),
                }],
            })
        return export_data

    def export_text(self) -> str:
        """导出全文文本"""
        novel = self._load_novel()
        lines = [f"《{novel.get('title', '群聊物语')}》", ""]

        # 出场人物
        characters = novel.get("characters", [])
        if characters:
            lines.append("【出场人物】")
            for char in characters:
                novel_name = char.get("novel_name", "")
                real_name = char.get("real_name", "")
                desc = char.get("description", "")
                if novel_name and real_name:
                    lines.append(f"  {novel_name}（{real_name}）：{desc}")
                elif novel_name:
                    lines.append(f"  {novel_name}：{desc}")
            lines.append("")

        # 简介（主题 + 自定义设定 + 剧情简介）
        synopsis_parts = []
        if novel.get("requirements"):
            synopsis_parts.append(f"风格：{novel['requirements']}")
        custom_settings = novel.get("custom_settings", [])
        if custom_settings:
            synopsis_parts.append("")
            synopsis_parts.append("用户设定：")
            for i, s in enumerate(custom_settings, 1):
                synopsis_parts.append(f"  {i}. {s.get('content', '')}")
        if novel.get("global_summary") and novel["global_summary"] != "故事尚未开始。":
            synopsis_parts.append(f"")
            synopsis_parts.append(f"剧情简介：{novel['global_summary']}")
        if synopsis_parts:
            lines.append("【简介】")
            lines.extend(synopsis_parts)
            lines.append("")

        for ch in novel.get("chapters", []):
            clean_title = self._strip_chapter_prefix(ch.get('title', ''))
            lines.append(f"第{ch.get('number', '?')}章 {clean_title}")
            lines.append("=" * 40)
            lines.append("")
            lines.append(self._strip_leading_chapter_heading(
                ch.get("content", ""), clean_title
            ))
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
            f"  章节预览：{'✅ 开启' if novel.get('preview_enabled', True) else '⏹ 关闭'}",
        ]
        if novel.get("global_summary"):
            lines.append(f"  故事进展：{truncate_text(novel['global_summary'], 200)}")
        next_direction = (novel.get("next_plot_direction") or "").strip()
        if next_direction:
            lines.append(f"  下一章走向：{truncate_text(next_direction, 120)}")
        story_bible = novel.get("story_bible", {})
        if isinstance(story_bible, dict) and story_bible.get("mainline"):
            lines.append(f"  主线目标：{truncate_text(story_bible['mainline'], 120)}")
        memories = novel.get("memory_entries", [])
        if memories:
            lines.append(f"  历史记忆：{len(memories)} 条")
        return "\n".join(lines)
