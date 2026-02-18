"""
知识库管理模块 — 世界观、人物、风格
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger

from .utils import PLUGIN_ID, generate_id, safe_json_load, safe_json_save, call_llm, parse_json_from_response
from .prompts import REFINE_WORLDVIEW_PROMPT


# =====================================================================
# 默认数据模板
# =====================================================================
_DEFAULT_WORLDVIEW: dict = {
    "name": "",
    "description": "",
    "rules": [],
    "locations": [],
    "history": [],
    "factions": [],
    "custom": {},
}

_DEFAULT_CHARACTERS: dict = {"characters": []}

_DEFAULT_STYLE: dict = {
    "name": "",
    "description": "",
    "samples": [],
    "guidelines": "",
    "active": True,
}


class KnowledgeBase:
    """统一管理三类知识库：世界观、人物、风格"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.kb_dir = data_dir / "knowledge"
        self.styles_dir = self.kb_dir / "styles"
        self._worldview_path = self.kb_dir / "worldview.json"
        self._characters_path = self.kb_dir / "characters.json"

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.styles_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 世界观
    # ------------------------------------------------------------------
    def load_worldview(self) -> dict:
        return safe_json_load(self._worldview_path, dict(_DEFAULT_WORLDVIEW))

    def save_worldview(self, data: dict) -> None:
        safe_json_save(self._worldview_path, data)

    def update_worldview(self, section: str, value: Any) -> dict:
        """更新世界观的某个字段"""
        wv = self.load_worldview()
        if section in ("rules", "locations", "history", "factions"):
            # 列表类字段 → 追加
            if isinstance(value, list):
                wv[section].extend(value)
            else:
                wv[section].append(value)
        else:
            wv[section] = value
        self.save_worldview(wv)
        return wv

    def delete_worldview_item(self, section: str, keyword: str) -> tuple[bool, str]:
        """
        从世界观的列表类字段中删除包含 keyword 的条目。
        对于字符串字段（name/description）则清空。
        返回 (是否成功, 消息)
        """
        wv = self.load_worldview()

        if section in ("name", "description"):
            if wv.get(section):
                wv[section] = ""
                self.save_worldview(wv)
                return True, f"已清空世界观的「{section}」"
            return False, f"世界观的「{section}」已经是空的"

        if section not in ("rules", "locations", "history", "factions"):
            return False, f"不支持的字段：{section}"

        items = wv.get(section, [])
        if not items:
            return False, f"世界观的「{section}」中没有任何条目"

        # 查找匹配的条目
        remaining = []
        removed = []
        for item in items:
            item_str = item if isinstance(item, str) else _json.dumps(item, ensure_ascii=False)
            if keyword in item_str:
                removed.append(item_str)
            else:
                remaining.append(item)

        if not removed:
            return False, f"未找到包含「{keyword}」的条目"

        wv[section] = remaining
        self.save_worldview(wv)
        return True, f"已删除 {len(removed)} 条：{'、'.join(removed[:3])}"

    def clear_worldview(self) -> None:
        """清空整个世界观"""
        self.save_worldview(dict(_DEFAULT_WORLDVIEW))
        logger.info(f"[{PLUGIN_ID}] 世界观已清空")

    def get_worldview_summary(self, max_len: int = 800) -> str:
        """获取世界观摘要文本（用于 prompt）"""
        wv = self.load_worldview()
        parts = []
        if wv.get("name"):
            parts.append(f"世界名称：{wv['name']}")
        if wv.get("description"):
            parts.append(f"描述：{wv['description']}")
        if wv.get("rules"):
            parts.append("规则/法则：" + "；".join(wv["rules"][:10]))
        if wv.get("locations"):
            locs = [loc if isinstance(loc, str) else loc.get("name", "") for loc in wv["locations"][:10]]
            parts.append("地点：" + "、".join(locs))
        if wv.get("factions"):
            facs = [f if isinstance(f, str) else f.get("name", "") for f in wv["factions"][:10]]
            parts.append("势力：" + "、".join(facs))
        text = "\n".join(parts)
        return text[:max_len] if len(text) > max_len else text

    # ------------------------------------------------------------------
    # 人物
    # ------------------------------------------------------------------
    def load_characters(self) -> dict:
        return safe_json_load(self._characters_path, dict(_DEFAULT_CHARACTERS))

    def save_characters(self, data: dict) -> None:
        safe_json_save(self._characters_path, data)

    def add_character(self, name: str, description: str, **kwargs) -> dict:
        """添加角色，返回新角色数据。如果同名角色已存在则更新。"""
        # 重复检测：同名角色已存在时更新而非新增
        existing = self.get_character(name)
        if existing:
            updates = {}
            if description and description not in ("暂无描述", f"场景中新出现的角色"):
                updates["description"] = description
            for k, v in kwargs.items():
                if v:
                    updates[k] = v
            if updates:
                updated = self.update_character(existing["id"], updates)
                if updated:
                    logger.info(f"[{PLUGIN_ID}] 更新已有角色：{name} ({existing['id']})")
                    return updated
            return existing

        data = self.load_characters()
        char = {
            "id": generate_id("char"),
            "name": name,
            "aliases": kwargs.get("aliases", []),
            "description": description,
            "background": kwargs.get("background", ""),
            "abilities": kwargs.get("abilities", []),
            "relationships": kwargs.get("relationships", []),
            "status": kwargs.get("status", "alive"),
            "first_appearance": kwargs.get("first_appearance", ""),
            "notes": kwargs.get("notes", ""),
        }
        data["characters"].append(char)
        self.save_characters(data)
        logger.info(f"[{PLUGIN_ID}] 添加角色：{name} ({char['id']})")
        return char

    def update_character(self, char_id: str, updates: dict) -> Optional[dict]:
        """按 ID 更新角色字段"""
        data = self.load_characters()
        for ch in data["characters"]:
            if ch["id"] == char_id:
                ch.update(updates)
                self.save_characters(data)
                return ch
        return None

    def get_character(self, name_or_id: str) -> Optional[dict]:
        """通过名字或 ID 查找角色"""
        data = self.load_characters()
        for ch in data["characters"]:
            if ch["id"] == name_or_id or ch["name"] == name_or_id:
                return ch
            if name_or_id in ch.get("aliases", []):
                return ch
        return None

    def delete_character(self, name_or_id: str) -> tuple[bool, str]:
        """通过名字或 ID 删除角色。返回 (是否成功, 消息)"""
        data = self.load_characters()
        for i, ch in enumerate(data["characters"]):
            if ch["id"] == name_or_id or ch["name"] == name_or_id:
                removed = data["characters"].pop(i)
                self.save_characters(data)
                logger.info(f"[{PLUGIN_ID}] 删除角色：{removed['name']} ({removed['id']})")
                return True, f"已删除角色「{removed['name']}」"
            if name_or_id in ch.get("aliases", []):
                removed = data["characters"].pop(i)
                self.save_characters(data)
                logger.info(f"[{PLUGIN_ID}] 删除角色：{removed['name']} ({removed['id']})")
                return True, f"已删除角色「{removed['name']}」"
        return False, f"未找到角色「{name_or_id}」"

    def list_characters(self) -> list[dict]:
        return self.load_characters().get("characters", [])

    def get_characters_summary(self, char_ids: list[str] | None = None, max_len: int = 800) -> str:
        """获取人物摘要文本（用于 prompt）。如果指定 char_ids 则只返回相关角色。"""
        chars = self.list_characters()
        if char_ids:
            chars = [c for c in chars if c["id"] in char_ids or c["name"] in char_ids]
        if not chars:
            return "暂无角色设定"
        parts = []
        for c in chars[:20]:
            line = f"- {c['name']}：{c.get('description', '无描述')}"
            if c.get("abilities"):
                line += f"（能力：{'、'.join(c['abilities'][:5])}）"
            parts.append(line)
        text = "\n".join(parts)
        return text[:max_len] if len(text) > max_len else text

    # ------------------------------------------------------------------
    # 风格
    # ------------------------------------------------------------------
    def add_style(self, name: str, description: str = "", guidelines: str = "", samples: list[str] | None = None) -> dict:
        """添加一个新的写作风格"""
        style = dict(_DEFAULT_STYLE)
        style["name"] = name
        style["description"] = description
        style["guidelines"] = guidelines
        style["samples"] = samples or []
        path = self.styles_dir / f"{name}.json"
        safe_json_save(path, style)
        logger.info(f"[{PLUGIN_ID}] 添加风格：{name}")
        return style

    def get_style(self, name: str) -> Optional[dict]:
        path = self.styles_dir / f"{name}.json"
        if not path.exists():
            return None
        return safe_json_load(path)

    def update_style(self, name: str, updates: dict) -> Optional[dict]:
        style = self.get_style(name)
        if not style:
            return None
        style.update(updates)
        safe_json_save(self.styles_dir / f"{name}.json", style)
        return style

    def add_style_sample(self, name: str, sample: str) -> bool:
        """向指定风格追加一段示例文本"""
        style = self.get_style(name)
        if not style:
            return False
        style["samples"].append(sample)
        safe_json_save(self.styles_dir / f"{name}.json", style)
        return True

    def list_styles(self) -> list[dict]:
        """列出所有风格"""
        styles = []
        if not self.styles_dir.exists():
            return styles
        for p in sorted(self.styles_dir.glob("*.json")):
            s = safe_json_load(p)
            if s:
                styles.append(s)
        return styles

    # ------------------------------------------------------------------
    # 综合上下文构建
    # ------------------------------------------------------------------
    def get_context_for_scene(self, char_ids: list[str] | None = None) -> dict:
        """
        构建写作场景所需的知识库上下文（用于注入 prompt）。
        返回 dict 包含 worldview_summary, characters_info 等。
        """
        return {
            "worldview_summary": self.get_worldview_summary(),
            "worldview_full": self.load_worldview(),
            "characters_info": self.get_characters_summary(char_ids),
            "characters_full": self.list_characters(),
        }

    def search(self, query: str) -> list[dict]:
        """简单关键词搜索知识库内容"""
        results = []
        query_lower = query.lower()

        # 搜索世界观
        wv = self.load_worldview()
        wv_text = str(wv).lower()
        if query_lower in wv_text:
            results.append({"type": "worldview", "match": "世界观中包含相关内容"})

        # 搜索角色
        for ch in self.list_characters():
            ch_text = str(ch).lower()
            if query_lower in ch_text:
                results.append({"type": "character", "name": ch["name"], "id": ch["id"]})

        # 搜索风格
        for s in self.list_styles():
            s_text = str(s).lower()
            if query_lower in s_text:
                results.append({"type": "style", "name": s["name"]})

        return results

    # ------------------------------------------------------------------
    # AI 整理完善世界观
    # ------------------------------------------------------------------
    async def refine_worldview_with_ai(
        self,
        provider,
        recent_ideas: str = "",
        story_progress: str = "",
    ) -> dict:
        """
        使用 AI 整理和完善世界观文档。
        将散乱的设定信息结构化，补充合理细节，标注矛盾。
        返回整理后的世界观 dict。
        """
        current_wv = self.load_worldview()
        chars_summary = self.get_characters_summary()

        prompt = REFINE_WORLDVIEW_PROMPT.format(
            current_worldview=_json.dumps(current_wv, ensure_ascii=False, indent=2)[:3000],
            characters=chars_summary[:1500],
            recent_ideas=recent_ideas or "暂无新创意",
            story_progress=story_progress or "故事尚未开始",
        )

        try:
            response = await call_llm(provider, prompt, timeout=120)
            result = parse_json_from_response(response)
            if result and isinstance(result, dict):
                # 保留旧世界观的备份
                backup_path = self.kb_dir / "worldview_backup.json"
                safe_json_save(backup_path, current_wv)

                # 合并：AI 整理的结果 + 保留已有的额外字段
                refined = dict(_DEFAULT_WORLDVIEW)
                for key in refined:
                    if key in result and result[key]:
                        refined[key] = result[key]
                    elif key in current_wv and current_wv[key]:
                        refined[key] = current_wv[key]

                # 保存 notes 到 custom 中
                if result.get("notes"):
                    refined.setdefault("custom", {})
                    refined["custom"]["ai_notes"] = result["notes"]

                self.save_worldview(refined)
                logger.info(f"[{PLUGIN_ID}] AI 世界观整理完成")
                return refined
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] AI 世界观整理失败: {e}")

        return current_wv

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空所有知识库数据"""
        import shutil
        if self.kb_dir.exists():
            shutil.rmtree(self.kb_dir)
        self.ensure_dirs()
        logger.info(f"[{PLUGIN_ID}] 知识库已重置")
