"""
小说生成引擎 — 场景级写作、多AI修正、摘要管理
"""
from __future__ import annotations

import json as _json
import re
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
    GENERATE_SCENE_PROMPT,
    REVISE_SCENE_PROMPT_PASS1,
    REVISE_SCENE_PROMPT_PASS2,
    REVISE_SCENE_PROMPT_PASS3,
    SUMMARIZE_SCENE_PROMPT,
    SUMMARIZE_GLOBAL_PROMPT,
    USER_GUIDED_REVISION_PROMPT,
    EXTRACT_NEW_CHARACTERS_PROMPT,
)
from .knowledge_base import KnowledgeBase


# =====================================================================
# 默认数据模板
# =====================================================================
_DEFAULT_NOVEL: dict = {
    "title": "",
    "synopsis": "",
    "current_style": "",
    "chapters": [],
    "global_summary": "故事尚未开始。",
    "contributors": [],
}


class NovelEngine:
    """小说生成引擎"""

    def __init__(self, data_dir: Path, kb: KnowledgeBase):
        self.data_dir = data_dir
        self.kb = kb
        self._path = data_dir / "novel.json"

    # ------------------------------------------------------------------
    # 数据读写
    # ------------------------------------------------------------------
    def _load(self) -> dict:
        return safe_json_load(self._path, dict(_DEFAULT_NOVEL))

    def _save(self, data: dict) -> None:
        safe_json_save(self._path, data)

    def is_initialized(self) -> bool:
        novel = self._load()
        return bool(novel.get("title"))

    def initialize(self, title: str, synopsis: str = "") -> dict:
        """初始化一部新小说"""
        novel = dict(_DEFAULT_NOVEL)
        novel["title"] = title
        novel["synopsis"] = synopsis
        self._save(novel)
        logger.info(f"[{PLUGIN_ID}] 小说初始化：{title}")
        return novel

    def get_novel(self) -> dict:
        return self._load()

    def add_contributor(self, name: str) -> None:
        """去重添加参与创作的群友昵称"""
        novel = self._load()
        contributors = novel.get("contributors", [])
        if name and name not in contributors:
            contributors.append(name)
            novel["contributors"] = contributors
            self._save(novel)

    def set_style(self, style_name: str) -> bool:
        """设定当前写作风格"""
        style = self.kb.get_style(style_name)
        if not style:
            return False
        novel = self._load()
        novel["current_style"] = style_name
        self._save(novel)
        return True

    # ------------------------------------------------------------------
    # 章节管理
    # ------------------------------------------------------------------
    def add_chapter(self, title: str) -> dict:
        """添加新章节"""
        novel = self._load()
        ch_num = len(novel["chapters"]) + 1
        chapter = {
            "id": generate_id("ch"),
            "title": title,
            "number": ch_num,
            "summary": "",
            "scenes": [],
        }
        novel["chapters"].append(chapter)
        self._save(novel)
        logger.info(f"[{PLUGIN_ID}] 新章节：第{ch_num}章 {title}")
        return chapter

    def get_chapter(self, chapter_id: str) -> Optional[dict]:
        novel = self._load()
        for ch in novel["chapters"]:
            if ch["id"] == chapter_id:
                return ch
        return None

    def get_current_chapter(self) -> Optional[dict]:
        """获取最后（当前）章节"""
        novel = self._load()
        if novel["chapters"]:
            return novel["chapters"][-1]
        return None

    def get_outline(self) -> str:
        """获取当前小说大纲"""
        novel = self._load()
        if not novel["chapters"]:
            return "📖 暂无章节"
        lines = [f"📖 《{novel['title']}》 大纲", ""]
        for ch in novel["chapters"]:
            lines.append(f"  第{ch.get('number', '?')}章 {ch['title']}")
            for sc in ch.get("scenes", []):
                status_icon = {"draft": "📝", "revised": "✏️", "final": "✅"}.get(
                    sc.get("status", "draft"), "📝"
                )
                lines.append(f"    {status_icon} {sc.get('title', '未命名场景')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 场景生成
    # ------------------------------------------------------------------
    async def generate_scene(
        self,
        scene_outline: str,
        provider,
        chapter_id: str = "",
        characters: list[str] | None = None,
        location: str = "",
        ideas_used: list[str] | None = None,
        max_length: int = 2000,
        search_context: str = "",
    ) -> Optional[dict]:
        """
        生成一个新场景。
        scene_outline: 场景大纲/要求
        """
        novel = self._load()
        if not novel["chapters"]:
            return None

        # 确定章节
        chapter = None
        if chapter_id:
            for ch in novel["chapters"]:
                if ch["id"] == chapter_id:
                    chapter = ch
                    break
        else:
            chapter = novel["chapters"][-1]
        if not chapter:
            return None

        # 获取风格
        style_name = novel.get("current_style", "")
        style = self.kb.get_style(style_name) if style_name else None
        style_guidelines = style.get("guidelines", "无特殊要求") if style else "无特殊要求"
        style_samples = ""
        if style and style.get("samples"):
            style_samples = "\n---\n".join(style["samples"][:3])
        else:
            style_samples = "无参考样本"

        # 获取知识库上下文
        kb_ctx = self.kb.get_context_for_scene(characters)

        # 获取前一场景摘要
        prev_summary = "这是本章的第一个场景。"
        if chapter["scenes"]:
            last_scene = chapter["scenes"][-1]
            prev_summary = last_scene.get("summary", "无摘要")

        # 构建地点信息
        location_info = location or "未指定"

        # 网络搜索上下文（预留）
        extra_context = ""
        if search_context:
            extra_context = f"\n## 网络搜索参考\n{search_context}"

        prompt = GENERATE_SCENE_PROMPT.format(
            novel_title=novel["title"],
            chapter_title=chapter["title"],
            global_summary=novel.get("global_summary", "暂无"),
            previous_scene_summary=prev_summary,
            characters_info=kb_ctx["characters_info"],
            location_info=location_info,
            scene_outline=scene_outline + extra_context,
            worldview_context=kb_ctx["worldview_summary"],
            style_name=style_name or "默认",
            style_guidelines=style_guidelines,
            style_samples=style_samples,
            max_length=max_length,
        )

        try:
            content = await call_llm(provider, prompt, timeout=180)
            if not content.strip():
                logger.error(f"[{PLUGIN_ID}] AI 生成空内容")
                return None

            # 生成场景摘要
            summary = await self._summarize_scene(provider, content)

            scene = {
                "id": generate_id("scene"),
                "title": scene_outline[:30],
                "content": content.strip(),
                "summary": summary,
                "characters_involved": characters or [],
                "location": location,
                "ideas_used": ideas_used or [],
                "version": 1,
                "revisions": [],
                "status": "draft",
            }

            chapter["scenes"].append(scene)

            # 更新全局摘要
            await self._update_global_summary(provider, novel, summary)

            self._save(novel)
            logger.info(f"[{PLUGIN_ID}] 场景生成完成：{scene['id']}")

            # 自动提取并添加新角色
            try:
                await self._extract_and_add_characters(provider, content.strip())
            except Exception as ex:
                logger.warning(f"[{PLUGIN_ID}] 自动提取新角色失败: {ex}")

            return scene

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 场景生成失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 自动提取新角色
    # ------------------------------------------------------------------
    async def _extract_and_add_characters(self, provider, scene_content: str) -> None:
        """从场景中提取新角色并写入人物库"""
        existing = self.kb.list_characters()
        existing_names = [c.get("name", "") for c in existing]
        # 别名也加入已知名单
        for c in existing:
            existing_names.extend(c.get("aliases", []))

        existing_text = ", ".join(existing_names) if existing_names else "暂无角色"

        prompt = EXTRACT_NEW_CHARACTERS_PROMPT.format(
            existing_characters=existing_text,
            scene_content=scene_content[:3000],
        )
        response = await call_llm(provider, prompt, timeout=60)
        result = parse_json_from_response(response)
        if not result:
            return

        new_chars = result.get("new_characters", [])
        for ch in new_chars:
            name = ch.get("name", "").strip()
            if not name or name in existing_names:
                continue
            desc = ch.get("description", "").strip()
            bg = ch.get("background", "").strip()
            self.kb.add_character(name, desc or f"场景中新出现的角色", background=bg)
            existing_names.append(name)
            logger.info(f"[{PLUGIN_ID}] 自动添加新角色：{name}")

    # ------------------------------------------------------------------
    # 多 AI 修正
    # ------------------------------------------------------------------
    async def revise_scene(
        self,
        scene_id: str,
        provider,
    ) -> Optional[dict]:
        """
        三轮修正：审读 → 修改 → 审校
        返回修正后的场景数据
        """
        novel = self._load()
        scene = None
        chapter = None
        for ch in novel["chapters"]:
            for sc in ch["scenes"]:
                if sc["id"] == scene_id:
                    scene = sc
                    chapter = ch
                    break
            if scene:
                break
        if not scene:
            return None

        original_content = scene["content"]
        style_name = novel.get("current_style", "")
        style = self.kb.get_style(style_name) if style_name else None
        kb_ctx = self.kb.get_context_for_scene(scene.get("characters_involved"))

        # —— 第一轮：审读 ——
        logger.info(f"[{PLUGIN_ID}] 修正 Pass 1：审读 {scene_id}")
        pass1_prompt = REVISE_SCENE_PROMPT_PASS1.format(
            novel_title=novel["title"],
            style_name=style_name or "默认",
            global_summary=novel.get("global_summary", "暂无"),
            scene_content=original_content,
            characters_info=kb_ctx["characters_info"],
        )
        try:
            pass1_response = await call_llm(provider, pass1_prompt, timeout=120)
            suggestions = parse_json_from_response(pass1_response)
            if not suggestions:
                suggestions = {"suggestions": [{"type": "通用", "fix": pass1_response[:500]}], "overall_comment": ""}
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 修正 Pass 1 失败: {e}")
            return None

        # —— 第二轮：执行修改 ——
        logger.info(f"[{PLUGIN_ID}] 修正 Pass 2：执行修改 {scene_id}")
        style_guidelines = style.get("guidelines", "") if style else ""
        style_samples = "\n---\n".join(style.get("samples", [])[:2]) if style else ""

        pass2_prompt = REVISE_SCENE_PROMPT_PASS2.format(
            scene_content=original_content,
            revision_suggestions=_json.dumps(suggestions, ensure_ascii=False, indent=2)[:3000],
            style_name=style_name or "默认",
            style_guidelines=style_guidelines,
            style_samples=style_samples or "无参考样本",
        )
        try:
            revised_content = await call_llm(provider, pass2_prompt, timeout=180)
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 修正 Pass 2 失败: {e}")
            return None

        # —— 第三轮：最终审校 ——
        logger.info(f"[{PLUGIN_ID}] 修正 Pass 3：最终审校 {scene_id}")
        pass3_prompt = REVISE_SCENE_PROMPT_PASS3.format(
            revised_content=revised_content,
            original_content=original_content[:2000],
            worldview_context=kb_ctx["worldview_summary"],
            characters_info=kb_ctx["characters_info"],
        )
        try:
            final_content = await call_llm(provider, pass3_prompt, timeout=180)
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 修正 Pass 3 失败: {e}")
            final_content = revised_content

        # 保存修正记录
        scene["revisions"].append({
            "version": scene["version"],
            "content": original_content,
        })
        scene["version"] += 1
        scene["content"] = final_content.strip()
        scene["status"] = "revised"

        # 更新摘要
        new_summary = await self._summarize_scene(provider, final_content)
        scene["summary"] = new_summary
        await self._update_global_summary(provider, novel, new_summary)

        self._save(novel)
        logger.info(f"[{PLUGIN_ID}] 场景修正完成：{scene_id} v{scene['version']}")
        return scene

    # ------------------------------------------------------------------
    # 用户人工介入修正章节
    # ------------------------------------------------------------------
    async def revise_chapter_with_user_input(
        self,
        chapter_number: int,
        user_feedback: str,
        provider,
    ) -> Optional[dict]:
        """
        根据用户的修改意见，使用 AI 修正指定章节。
        user_feedback: 用户收集到的所有反馈（文本 + 图片描述）
        返回修正后的章节 dict，失败返回 None。
        """
        novel = self._load()
        chapter = None
        for ch in novel["chapters"]:
            if ch.get("number") == chapter_number:
                chapter = ch
                break
        if not chapter:
            return None

        # 拼接章节当前内容
        chapter_content_parts = []
        for sc in chapter.get("scenes", []):
            if sc.get("title"):
                chapter_content_parts.append(f"—— {sc['title']} ——")
            chapter_content_parts.append(sc.get("content", ""))
            chapter_content_parts.append("")
        chapter_content = "\n".join(chapter_content_parts)

        if not chapter_content.strip():
            return None

        # 获取上下文
        style_name = novel.get("current_style", "")
        kb_ctx = self.kb.get_context_for_scene()

        prompt = USER_GUIDED_REVISION_PROMPT.format(
            novel_title=novel["title"],
            style_name=style_name or "默认",
            worldview_context=kb_ctx["worldview_summary"],
            characters_info=kb_ctx["characters_info"],
            chapter_number=chapter_number,
            chapter_title=chapter["title"],
            chapter_content=chapter_content[:5000],
            user_feedback=user_feedback[:3000],
        )

        try:
            revised_content = await call_llm(provider, prompt, timeout=180)
            if not revised_content.strip():
                return None

            # 将修正后的内容替换到章节中
            # 保存旧版本
            for sc in chapter.get("scenes", []):
                sc.setdefault("revisions", []).append({
                    "version": sc.get("version", 1),
                    "content": sc.get("content", ""),
                    "revision_type": "user_guided",
                })
                sc["version"] = sc.get("version", 1) + 1

            # 如果只有一个场景，直接替换
            if len(chapter["scenes"]) == 1:
                chapter["scenes"][0]["content"] = revised_content.strip()
                chapter["scenes"][0]["status"] = "revised"
            else:
                # 多场景：AI 返回的是整章内容，按场景分隔符拆分
                # 尝试按 "——" 分隔符拆分
                parts = re.split(r"——\s*(.+?)\s*——", revised_content.strip())
                if len(parts) >= len(chapter["scenes"]) * 2 - 1:
                    # 成功拆分
                    scene_idx = 0
                    for i in range(0, len(parts), 2):
                        if scene_idx < len(chapter["scenes"]):
                            content_part = parts[i].strip()
                            if content_part:
                                chapter["scenes"][scene_idx]["content"] = content_part
                                chapter["scenes"][scene_idx]["status"] = "revised"
                            scene_idx += 1
                else:
                    # 无法拆分，将全部内容放入第一个场景
                    chapter["scenes"][0]["content"] = revised_content.strip()
                    chapter["scenes"][0]["status"] = "revised"

            # 更新摘要
            new_summary = await self._summarize_scene(provider, revised_content[:3000])
            chapter["summary"] = new_summary
            await self._update_global_summary(provider, novel, new_summary)

            self._save(novel)
            logger.info(f"[{PLUGIN_ID}] 用户介入修正完成：第{chapter_number}章")
            return chapter

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 用户介入修正失败: {e}")
            return None

    def get_chapter_by_number(self, chapter_number: int) -> Optional[dict]:
        """按章节号获取章节"""
        novel = self._load()
        for ch in novel["chapters"]:
            if ch.get("number") == chapter_number:
                return ch
        return None

    # ------------------------------------------------------------------
    # 最新场景
    # ------------------------------------------------------------------
    def get_latest_scene(self) -> Optional[dict]:
        """获取最新写入的场景"""
        novel = self._load()
        for ch in reversed(novel["chapters"]):
            if ch["scenes"]:
                return ch["scenes"][-1]
        return None

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export_novel(self) -> str:
        """导出完整小说文本"""
        novel = self._load()
        lines = [f"《{novel['title']}》", ""]
        if novel.get("synopsis"):
            lines.append(f"【简介】{novel['synopsis']}")
            lines.append("")

        for ch in novel["chapters"]:
            lines.append(f"第{ch.get('number', '?')}章 {ch['title']}")
            lines.append("=" * 40)
            lines.append("")
            for sc in ch.get("scenes", []):
                if sc.get("title"):
                    lines.append(f"—— {sc['title']} ——")
                    lines.append("")
                lines.append(sc.get("content", ""))
                lines.append("")
            lines.append("")
        return "\n".join(lines)

    def export_chapter(self, chapter_number: int) -> Optional[str]:
        """导出指定章节"""
        novel = self._load()
        for ch in novel["chapters"]:
            if ch.get("number") == chapter_number:
                lines = [f"第{ch['number']}章 {ch['title']}", "=" * 40, ""]
                for sc in ch.get("scenes", []):
                    if sc.get("title"):
                        lines.append(f"—— {sc['title']} ——")
                        lines.append("")
                    lines.append(sc.get("content", ""))
                    lines.append("")
                return "\n".join(lines)
        return None

    def get_status(self) -> str:
        """获取小说当前状态"""
        novel = self._load()
        if not novel.get("title"):
            return "📖 尚未初始化小说。请使用 /小说 初始化 <标题>"
        total_scenes = sum(len(ch.get("scenes", [])) for ch in novel["chapters"])
        total_chars = sum(
            len(sc.get("content", ""))
            for ch in novel["chapters"]
            for sc in ch.get("scenes", [])
        )
        lines = [
            f"📖 《{novel['title']}》",
            f"  当前风格：{novel.get('current_style') or '未设定'}",
            f"  章节数：{len(novel['chapters'])}",
            f"  场景数：{total_scenes}",
            f"  总字数：{total_chars}",
        ]
        if novel.get("global_summary"):
            lines.append(f"  故事进展：{truncate_text(novel['global_summary'], 200)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    async def _summarize_scene(self, provider, content: str) -> str:
        """AI 生成场景摘要"""
        prompt = SUMMARIZE_SCENE_PROMPT.format(scene_content=content[:3000])
        try:
            summary = await call_llm(provider, prompt, timeout=60)
            return summary.strip()[:200]
        except Exception:
            return content[:100] + "..."

    async def _update_global_summary(self, provider, novel: dict, new_scene_summary: str) -> None:
        """AI 更新全局摘要"""
        prompt = SUMMARIZE_GLOBAL_PROMPT.format(
            old_summary=novel.get("global_summary", "暂无"),
            new_scene_summary=new_scene_summary,
        )
        try:
            new_global = await call_llm(provider, prompt, timeout=60)
            novel["global_summary"] = new_global.strip()[:500]
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 全局摘要更新失败: {e}")
