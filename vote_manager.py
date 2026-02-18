"""
投票系统模块 — 管理冲突解决投票
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .utils import PLUGIN_ID, generate_id, safe_json_load, safe_json_save, format_timestamp


class VoteManager:
    """群内投票管理器"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._path = data_dir / "votes.json"

    def _load(self) -> dict:
        return safe_json_load(self._path, {"votes": []})

    def _save(self, data: dict) -> None:
        safe_json_save(self._path, data)

    def create_vote(
        self,
        description: str,
        options: list[dict],
        related_idea_id: str = "",
        duration_minutes: int = 30,
    ) -> dict:
        """
        创建一个新投票。
        options: [{"key": "A", "label": "选项描述"}, ...]
        """
        from datetime import datetime, timedelta

        now = datetime.now()
        vote = {
            "id": generate_id("vote"),
            "type": "conflict_resolution",
            "description": description,
            "options": options,
            "ballots": {},
            "status": "open",
            "result": None,
            "related_idea_id": related_idea_id,
            "created_at": now.isoformat(timespec="seconds"),
            "closes_at": (now + timedelta(minutes=duration_minutes)).isoformat(timespec="seconds"),
        }
        data = self._load()
        data["votes"].append(vote)
        self._save(data)
        logger.info(f"[{PLUGIN_ID}] 创建投票 {vote['id']}: {description}")
        return vote

    def cast_vote(self, vote_id: str, user_id: str, option_key: str) -> tuple[bool, str]:
        """
        用户投票。
        返回 (成功, 提示消息)
        """
        data = self._load()
        for v in data["votes"]:
            if v["id"] == vote_id:
                if v["status"] != "open":
                    return False, "投票已结束"
                valid_keys = {o["key"] for o in v["options"]}
                if option_key not in valid_keys:
                    return False, f"无效选项，可选：{', '.join(sorted(valid_keys))}"
                old = v["ballots"].get(user_id)
                v["ballots"][user_id] = option_key
                self._save(data)
                if old:
                    return True, f"已将投票从 {old} 改为 {option_key}"
                return True, f"投票成功：{option_key}"
        return False, "未找到该投票"

    def close_vote(self, vote_id: str) -> Optional[dict]:
        """
        关闭投票并统计结果。
        返回投票数据（含 result 字段），或 None。
        """
        data = self._load()
        for v in data["votes"]:
            if v["id"] == vote_id:
                if v["status"] == "closed":
                    return v
                # 统计
                tally: dict[str, int] = {}
                for opt in v["options"]:
                    tally[opt["key"]] = 0
                for _, choice in v["ballots"].items():
                    tally[choice] = tally.get(choice, 0) + 1
                # 找到得票最多的
                winner = max(tally, key=lambda k: tally[k]) if tally else None
                v["status"] = "closed"
                v["result"] = {
                    "tally": tally,
                    "winner": winner,
                    "winner_label": next(
                        (o["label"] for o in v["options"] if o["key"] == winner), ""
                    ),
                    "total_votes": len(v["ballots"]),
                }
                self._save(data)
                logger.info(f"[{PLUGIN_ID}] 投票 {vote_id} 关闭，结果：{winner}")
                return v
        return None

    def get_active_votes(self) -> list[dict]:
        """获取所有进行中的投票"""
        data = self._load()
        return [v for v in data["votes"] if v["status"] == "open"]

    def get_vote(self, vote_id: str) -> Optional[dict]:
        data = self._load()
        for v in data["votes"]:
            if v["id"] == vote_id:
                return v
        return None

    def get_latest_active_vote(self) -> Optional[dict]:
        """获取最新的进行中投票"""
        active = self.get_active_votes()
        return active[-1] if active else None

    def format_vote_message(self, vote: dict) -> str:
        """格式化投票消息，用于群内发送"""
        lines = [
            "📊 【投票】" + vote["description"],
            "",
        ]
        for opt in vote["options"]:
            count = sum(1 for v in vote["ballots"].values() if v == opt["key"])
            lines.append(f"  {opt['key']}. {opt['label']}  [{count}票]")
        lines.append("")
        lines.append(f"投票方式：发送 /小说 投票 <选项字母>")
        if vote["status"] == "open":
            lines.append(f"截止时间：{vote.get('closes_at', '未定')}")
        else:
            r = vote.get("result", {})
            lines.append(f"✅ 投票已结束！结果：{r.get('winner', '?')}. {r.get('winner_label', '')}")
        return "\n".join(lines)

    def auto_close_expired(self) -> list[dict]:
        """自动关闭已过期的投票，返回被关闭的投票列表"""
        from datetime import datetime

        now = datetime.now()
        closed = []
        data = self._load()
        for v in data["votes"]:
            if v["status"] != "open":
                continue
            try:
                closes_at = datetime.fromisoformat(v["closes_at"])
                if now >= closes_at:
                    result = self.close_vote(v["id"])
                    if result:
                        closed.append(result)
            except (ValueError, KeyError):
                continue
        return closed
