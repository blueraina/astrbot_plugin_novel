"""
创意收集与评分模块 — 群友创意管理
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .utils import (
    PLUGIN_ID,
    generate_id,
    safe_json_load,
    safe_json_save,
    format_timestamp,
    call_llm,
    parse_json_from_response,
)
from .prompts import SCORE_IDEA_PROMPT, CONFLICT_CHECK_PROMPT
from .knowledge_base import KnowledgeBase
from .vote_manager import VoteManager


class IdeaManager:
    """创意收集、评分、冲突检测"""

    def __init__(self, data_dir: Path, kb: KnowledgeBase, vm: VoteManager):
        self.data_dir = data_dir
        self.kb = kb
        self.vm = vm
        self._path = data_dir / "ideas.json"

    def _load(self) -> dict:
        return safe_json_load(self._path, {"ideas": []})

    def _save(self, data: dict) -> None:
        safe_json_save(self._path, data)

    # ------------------------------------------------------------------
    # 提交创意
    # ------------------------------------------------------------------
    def submit_idea(
        self,
        author: str,
        author_id: str,
        content: str,
        idea_type: str = "plot",
    ) -> dict:
        """提交一个新创意，状态为 pending"""
        idea = {
            "id": generate_id("idea"),
            "author": author,
            "author_id": author_id,
            "content": content,
            "type": idea_type,
            "timestamp": format_timestamp(),
            "scores": [],
            "weighted_avg": 0.0,
            "status": "pending",
            "conflict_info": None,
            "votes": {},
        }
        data = self._load()
        data["ideas"].append(idea)
        self._save(data)
        logger.info(f"[{PLUGIN_ID}] 新创意 {idea['id']} by {author}: {content[:50]}")
        return idea

    # ------------------------------------------------------------------
    # 多 AI 打分
    # ------------------------------------------------------------------
    async def score_idea(
        self,
        idea_id: str,
        providers: list,
        novel_title: str = "",
        novel_synopsis: str = "",
    ) -> Optional[dict]:
        """
        对创意进行多 AI 打分（每个 provider 评分一次），返回更新后的创意数据。
        providers: 评分用的 provider 列表（各自调用一次）
        """
        data = self._load()
        idea = None
        for i in data["ideas"]:
            if i["id"] == idea_id:
                idea = i
                break
        if not idea:
            return None

        # 构建 prompt 上下文
        wv_summary = self.kb.get_worldview_summary()
        approved = self.get_approved_ideas()
        existing_text = "\n".join(
            f"- [{a['type']}] {a['content'][:100]}" for a in approved[:20]
        ) or "暂无"

        scores = []
        for idx, prov in enumerate(providers, 1):
            # 提取模型名称
            model_name = "未知模型"
            try:
                if hasattr(prov, "model_name"):
                    model_name = prov.model_name or model_name
                elif hasattr(prov, "model"):
                    model_name = prov.model or model_name
                elif hasattr(prov, "origin"):
                    model_name = prov.origin or model_name
            except Exception:
                pass

            prompt = SCORE_IDEA_PROMPT.format(
                novel_title=novel_title or "未定",
                novel_synopsis=novel_synopsis or "暂无",
                worldview_summary=wv_summary,
                existing_ideas=existing_text,
                author=idea["author"],
                idea_type=idea["type"],
                idea_content=idea["content"],
            )
            try:
                response = await call_llm(prov, prompt)
                result = parse_json_from_response(response)
                if result and "overall" in result:
                    scores.append({
                        "ai_id": idx,
                        "model_name": model_name,
                        "score": result["overall"],
                        "originality": result.get("originality", 0),
                        "coherence": result.get("coherence", 0),
                        "narrative_value": result.get("narrative_value", 0),
                        "reason": result.get("reason", ""),
                    })
                    logger.info(
                        f"[{PLUGIN_ID}] 创意 {idea_id} AI-{idx}({model_name}) 打分: {result['overall']}"
                    )
                else:
                    logger.warning(f"[{PLUGIN_ID}] AI-{idx}({model_name}) 打分解析失败")
            except Exception as e:
                logger.error(f"[{PLUGIN_ID}] AI-{idx}({model_name}) 打分出错: {e}")

        if scores:
            avg = sum(s["score"] for s in scores) / len(scores)
            idea["scores"] = scores
            idea["weighted_avg"] = round(avg, 1)
        else:
            idea["weighted_avg"] = 0.0

        self._save(data)
        return idea

    # ------------------------------------------------------------------
    # 冲突检测
    # ------------------------------------------------------------------
    async def check_conflict(
        self,
        idea_id: str,
        provider,
    ) -> Optional[dict]:
        """
        检测创意是否与现有设定冲突。
        返回 {"has_conflict": bool, "conflicts": [...], "suggestion": "..."}
        """
        data = self._load()
        idea = None
        for i in data["ideas"]:
            if i["id"] == idea_id:
                idea = i
                break
        if not idea:
            return None

        import json as _json

        wv = self.kb.load_worldview()
        chars = self.kb.list_characters()
        approved = self.get_approved_ideas()

        prompt = CONFLICT_CHECK_PROMPT.format(
            worldview=_json.dumps(wv, ensure_ascii=False, indent=2)[:2000],
            characters=self.kb.get_characters_summary()[:1000],
            approved_ideas="\n".join(
                f"- [{a['type']}] {a['content'][:100]}" for a in approved[:15]
            ) or "暂无",
            new_idea=f"[{idea['type']}] {idea['content']}",
        )

        try:
            response = await call_llm(provider, prompt)
            result = parse_json_from_response(response)
            if result:
                idea["conflict_info"] = result
                if result.get("has_conflict"):
                    idea["status"] = "conflict"
                self._save(data)
                return result
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 冲突检测失败: {e}")

        return {"has_conflict": False, "conflicts": [], "suggestion": "检测失败，默认无冲突"}

    # ------------------------------------------------------------------
    # 创意投票处理
    # ------------------------------------------------------------------
    def create_conflict_vote(self, idea_id: str, conflict_info: dict, duration_minutes: int = 30) -> Optional[dict]:
        """为冲突的创意创建投票"""
        data = self._load()
        idea = None
        for i in data["ideas"]:
            if i["id"] == idea_id:
                idea = i
                break
        if not idea:
            return None

        conflicts_desc = "; ".join(
            c.get("description", "") for c in conflict_info.get("conflicts", [])
        )
        suggestion = conflict_info.get("suggestion", "无建议")

        options = [
            {"key": "A", "label": f"采用新创意：{idea['content'][:60]}"},
            {"key": "B", "label": "保留旧设定，拒绝此创意"},
            {"key": "C", "label": f"折中方案：{suggestion[:80]}"},
        ]

        vote = self.vm.create_vote(
            description=f"创意冲突：{conflicts_desc[:100]}",
            options=options,
            related_idea_id=idea_id,
            duration_minutes=duration_minutes,
        )
        return vote

    def apply_vote_result(self, vote: dict) -> str:
        """根据投票结果处理创意"""
        result = vote.get("result", {})
        winner = result.get("winner")
        idea_id = vote.get("related_idea_id")
        if not idea_id:
            return "投票未关联创意"

        data = self._load()
        idea = None
        for i in data["ideas"]:
            if i["id"] == idea_id:
                idea = i
                break
        if not idea:
            return "未找到关联创意"

        if winner == "A":
            idea["status"] = "approved"
            self._save(data)
            return f"✅ 创意已采纳：{idea['content'][:50]}"
        elif winner == "B":
            idea["status"] = "rejected"
            self._save(data)
            return f"❌ 创意已拒绝：{idea['content'][:50]}"
        elif winner == "C":
            idea["status"] = "approved"
            idea["content"] += f"\n[折中修改] {vote['options'][2]['label']}"
            self._save(data)
            return f"🔄 采用折中方案"
        return "未知投票结果"

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_idea(self, idea_id: str) -> Optional[dict]:
        data = self._load()
        for i in data["ideas"]:
            if i["id"] == idea_id:
                return i
        return None

    def get_pending_ideas(self) -> list[dict]:
        return [i for i in self._load()["ideas"] if i["status"] == "pending"]

    def get_approved_ideas(self) -> list[dict]:
        return [i for i in self._load()["ideas"] if i["status"] == "approved"]

    def get_all_ideas(self) -> list[dict]:
        return self._load()["ideas"]

    def approve_idea(self, idea_id: str) -> bool:
        data = self._load()
        for i in data["ideas"]:
            if i["id"] == idea_id:
                i["status"] = "approved"
                self._save(data)
                return True
        return False

    def reject_idea(self, idea_id: str) -> bool:
        data = self._load()
        for i in data["ideas"]:
            if i["id"] == idea_id:
                i["status"] = "rejected"
                self._save(data)
                return True
        return False
