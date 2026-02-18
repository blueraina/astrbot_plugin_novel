\
from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

# 文件发送组件导入（兼容不同版本）
try:
    from astrbot.api.message_components import Plain, File as FileComp
except ImportError:
    try:
        from astrbot.core.message.components import Plain, File as FileComp
    except ImportError:
        Plain = None
        FileComp = None

try:
    from astrbot.api import AstrBotConfig  # type: ignore
except Exception:  # pragma: no cover
    AstrBotConfig = Any  # type: ignore

from .utils import PLUGIN_ID, truncate_text
from .knowledge_base import KnowledgeBase
from .idea_manager import IdeaManager
from .novel_engine import NovelEngine
from .vote_manager import VoteManager
from .exporter import export_txt, export_epub, export_pdf
from .chat_novel import ChatNovelEngine


def _resolve_data_dir(plugin_name: str) -> Path:
    return Path(get_astrbot_data_path()) / "plugin_data" / (plugin_name or PLUGIN_ID)


# =====================================================================
# 每个群的上下文（数据隔离）
# =====================================================================
@dataclass
class GroupContext:
    """单个群的完整运行上下文"""
    group_id: str
    data_dir: Path
    kb: KnowledgeBase = field(init=False)
    ideas: IdeaManager = field(init=False)
    engine: NovelEngine = field(init=False)
    votes: VoteManager = field(init=False)
    chat_novel: ChatNovelEngine = field(init=False)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kb = KnowledgeBase(self.data_dir)
        self.kb.ensure_dirs()
        self.votes = VoteManager(self.data_dir)
        self.ideas = IdeaManager(self.data_dir, self.kb, self.votes)
        self.engine = NovelEngine(self.data_dir, self.kb)
        self.chat_novel = ChatNovelEngine(self.data_dir)

    def reset_all(self) -> None:
        """清空该群的所有小说数据"""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kb = KnowledgeBase(self.data_dir)
        self.kb.ensure_dirs()
        self.votes = VoteManager(self.data_dir)
        self.ideas = IdeaManager(self.data_dir, self.kb, self.votes)
        self.engine = NovelEngine(self.data_dir, self.kb)
        self.chat_novel = ChatNovelEngine(self.data_dir)


# =====================================================================
# 帮助信息
# =====================================================================
HELP_TEXT = """\
📖 【群体协作小说插件】指令一览

▸ /小说 帮助          显示本帮助
▸ /小说 初始化 <标题>  开始一部新小说
▸ /小说 状态          查看当前进度
▸ /小说 重置          ⚠️ 清空所有数据重新开始

◈ 知识库
▸ /小说 世界观        查看世界观
▸ /小说 设定 <内容>   添加世界观设定
▸ /小说 整理世界观    AI 整理完善世界观
▸ /小说 删除设定 <字段> <关键词>  删除世界观条目
▸ /小说 清空世界观    清空整个世界观
▸ /小说 添加人物 <名字> <描述>
▸ /小说 修改人物 <名字> <内容>  修改角色设定
▸ /小说 删除人物 <名字>  删除角色
▸ /小说 人物列表      所有角色
▸ /小说 人物 <名字>   角色详情

◈ 风格
▸ /小说 风格列表      可用风格
▸ /小说 添加风格 <名称>  创建新风格（随后发送示例文本）
▸ /小说 风格样本 <名称> <文本>  追加风格示例
▸ /小说 切换风格 <名称>  切换写作风格

◈ 创意
▸ /小说 创意 <内容>   提交创意（自动打分+冲突检测）
▸ /小说 强制创意 <内容>  跳过评分直接采纳
▸ /小说 强制采纳      强制通过最近被拒的创意
▸ /小说 创意列表      已采纳的创意

◈ 写作
▸ /小说 新章节 <标题>  开始新章节
▸ /小说 写 <场景描述>  AI 生成新场景
▸ /小说 修正          多 AI 修正最新场景
▸ /小说 大纲          查看大纲
▸ /小说 更改 <章节号> <描述>  用户介入修正章节
▸ /小说 更改 <章节号> 开始  进入交互修正模式
▸ /小说 结束更改        结束交互修正并提交 AI 修改

◈ 投票
▸ /小说 投票 <选项>   对当前投票投票

◈ 导出
▸ /小说 导出 [格式]   导出全文（格式：txt/epub/pdf）
▸ /小说 阅读 [章节号]  阅读章节
"""


@register(
    PLUGIN_ID,
    "blueraina",
    "群体协作长篇小说插件 — 群友创意 + AI 写作（每群独立/知识库/多AI打分/冲突投票/风格模仿/用户修正/EPUB/PDF导出）",
    "2.4.0",
    "https://github.com/blueraina/astrbot_plugin_novel",
)
class NovelPlugin(Star):
    """群体协作长篇小说 AstrBot 插件"""

    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.context = context
        self.config = config

        plugin_name = getattr(self, "name", PLUGIN_ID) or PLUGIN_ID
        self.base_data_dir = _resolve_data_dir(plugin_name)

        # 每群上下文 {group_id: GroupContext}
        self._groups: dict[str, GroupContext] = {}

        # 风格添加会话状态 {group_id: style_name}
        self._pending_style: dict[str, str] = {}

        # 用户介入修正状态 {group_id: {"chapter_num": int, "messages": [str]}}
        self._pending_revision: dict[str, dict] = {}

        # 世界观整理计数器 {group_id: counter}
        # 每发生 N 次关键操作后自动整理
        self._wv_refine_counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{PLUGIN_ID}] 插件初始化完成，base_data_dir={self.base_data_dir}")

    async def terminate(self) -> None:
        logger.info(f"[{PLUGIN_ID}] 插件已卸载")

    # ------------------------------------------------------------------
    # 每群数据隔离
    # ------------------------------------------------------------------
    def _get_group_ctx(self, group_id: str) -> GroupContext:
        """获取或创建群上下文（懒加载）"""
        if group_id not in self._groups:
            group_dir = self.base_data_dir / "groups" / group_id
            self._groups[group_id] = GroupContext(group_id=group_id, data_dir=group_dir)
            logger.info(f"[{PLUGIN_ID}] 初始化群上下文：{group_id}")
        return self._groups[group_id]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _cfg(self, k: str, default: Any = None) -> Any:
        if not self.config:
            return default
        try:
            return self.config.get(k, default)
        except Exception:
            return default

    def _cfg_int(self, k: str, default: int) -> int:
        v = self._cfg(k, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, k: str, default: bool) -> bool:
        v = self._cfg(k, default)
        try:
            return bool(v)
        except Exception:
            return default

    def _allow(self, event: AstrMessageEvent) -> bool:
        """检查是否允许执行指令"""
        gid = event.get_group_id()
        if not gid:
            return self._cfg_bool("allow_private_commands", False)

        # 检查群白名单
        enabled = self._cfg("enabled_groups", [])
        if enabled and isinstance(enabled, list) and len(enabled) > 0:
            return str(gid) in [str(g) for g in enabled]
        # 如果未配置白名单，则所有群都允许
        return True

    def _get_provider(self):
        """获取当前 LLM provider"""
        return self.context.get_using_provider()

    def _get_provider_for(self, role: str):
        """按功能角色获取 LLM provider，如未配置则回退到全局默认"""
        cfg_key = f"provider_{role}"
        provider_id = self._cfg(cfg_key, "")
        if provider_id:
            try:
                p = self.context.get_provider_by_id(provider_id)
                if p:
                    return p
                logger.warning(f"[{PLUGIN_ID}] 配置的 {cfg_key}={provider_id} 无效，回退到默认")
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] 获取 provider {provider_id} 失败: {e}，回退到默认")
        return self.context.get_using_provider()

    def _get_ctx(self, event: AstrMessageEvent) -> Optional[GroupContext]:
        """从事件中获取群上下文"""
        gid = event.get_group_id()
        if not gid:
            return None
        return self._get_group_ctx(gid)

    def _record_contributor(self, ctx: GroupContext, event: AstrMessageEvent) -> None:
        """记录参与创作的群友昵称到 novel.json"""
        if not ctx.engine.is_initialized():
            return
        name = event.get_sender_name() if hasattr(event, 'get_sender_name') else None
        if not name:
            name = event.get_sender_id() or "unknown"
        ctx.engine.add_contributor(str(name))

    async def _maybe_refine_worldview(self, ctx: GroupContext) -> None:
        """在关键操作后自动整理世界观（每5次操作触发一次）"""
        gid = ctx.group_id
        self._wv_refine_counter[gid] = self._wv_refine_counter.get(gid, 0) + 1
        if self._wv_refine_counter[gid] >= 5:
            self._wv_refine_counter[gid] = 0
            try:
                provider = self._get_provider_for("worldview")
                novel = ctx.engine.get_novel()
                ideas_data = ctx.ideas.get_approved_ideas()
                recent_ideas = "\n".join([
                    f"- {i.get('content', '')}" for i in ideas_data[-5:]
                ]) if ideas_data else ""
                story_progress = novel.get("global_summary", "") if novel else ""

                asyncio.create_task(
                    ctx.kb.refine_worldview_with_ai(
                        provider,
                        recent_ideas=recent_ideas,
                        story_progress=story_progress,
                    )
                )
                logger.info(f"[{PLUGIN_ID}] 触发异步世界观整理（群 {gid}）")
            except Exception as e:
                logger.error(f"[{PLUGIN_ID}] 世界观自动整理触发失败: {e}")

    # ------------------------------------------------------------------
    # 指令组: /小说
    # ------------------------------------------------------------------
    @filter.command_group("小说", alias={"novel"})
    def novel(self):
        pass

    # ====== 基础指令 ======

    @novel.command("帮助", alias={"help"})
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助"""
        if not self._allow(event):
            yield event.plain_result("本群未开启小说插件。")
            return
        yield event.plain_result(HELP_TEXT)

    @novel.command("初始化", alias={"init"})
    async def cmd_init(self, event: AstrMessageEvent, title: str = ""):
        """初始化小说"""
        if not self._allow(event):
            yield event.plain_result("本群未开启小说插件。")
            return
        if not title:
            yield event.plain_result("请指定小说标题：/小说 初始化 <标题>")
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if ctx.engine.is_initialized():
            novel = ctx.engine.get_novel()
            yield event.plain_result(
                f"⚠️ 已有小说《{novel.get('title', '')}》。如需重新开始，请先执行 /小说 重置"
            )
            return
        ctx.engine.initialize(title)
        yield event.plain_result(
            f"✅ 小说《{title}》已创建！\n"
            f"📌 请先添加世界观设定和角色，再开始写作。\n"
            f"使用 /小说 帮助 查看所有指令。"
        )

    @novel.command("状态", alias={"status"})
    async def cmd_status(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx or not ctx.engine.is_initialized():
            yield event.plain_result("尚未初始化小说。请使用 /小说 初始化 <标题>")
            return
        yield event.plain_result(ctx.engine.get_status())

    # ====== 重置指令 ======

    @novel.command("重置", alias={"reset"})
    async def cmd_reset(self, event: AstrMessageEvent):
        """清空所有数据，重新开始"""
        if not self._allow(event):
            yield event.plain_result("本群未开启小说插件。")
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        ctx.reset_all()
        # 清除相关缓存
        gid = ctx.group_id
        self._pending_style.pop(gid, None)
        self._wv_refine_counter.pop(gid, None)
        yield event.plain_result(
            "✅ 已清空所有小说数据（知识库、创意、章节、投票）。\n"
            "现在可以使用 /小说 初始化 <标题> 开始新故事！"
        )

    # ====== 知识库指令 ======

    @novel.command("世界观", alias={"worldview"})
    async def cmd_worldview(self, event: AstrMessageEvent):
        """查看世界观"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        wv = ctx.kb.load_worldview()
        parts = ["🌍 当前世界观"]
        if wv.get("name"):
            parts.append(f"名称：{wv['name']}")
        if wv.get("description"):
            parts.append(f"描述：{wv['description']}")
        if wv.get("rules"):
            parts.append(f"规则：{'; '.join(str(r) for r in wv['rules'][:10])}")
        if wv.get("locations"):
            locs = []
            for loc in wv["locations"][:10]:
                if isinstance(loc, dict):
                    locs.append(loc.get("name", str(loc)))
                else:
                    locs.append(str(loc))
            parts.append(f"地点：{'、'.join(locs)}")
        if wv.get("factions"):
            facs = []
            for f in wv["factions"][:10]:
                if isinstance(f, dict):
                    facs.append(f.get("name", str(f)))
                else:
                    facs.append(str(f))
            parts.append(f"势力：{'、'.join(facs)}")
        if wv.get("history"):
            parts.append(f"历史事件数：{len(wv['history'])}")
        notes = wv.get("custom", {}).get("ai_notes", "")
        if notes:
            parts.append(f"📋 AI 整理备注：{truncate_text(notes, 200)}")
        if len(parts) == 1:
            parts.append("尚未设定。使用 /小说 设定 <内容> 添加。")
        yield event.plain_result("\n".join(parts))

    @novel.command("设定", alias={"setting"})
    async def cmd_setting(self, event: AstrMessageEvent, text: str = ""):
        """添加/修改世界观设定"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        content = text.strip()
        if not content:
            yield event.plain_result("请输入设定内容：/小说 设定 <内容>")
            return
        # 尝试解析 key: value 格式
        if "：" in content or ":" in content:
            sep = "：" if "：" in content else ":"
            key, _, val = content.partition(sep)
            key = key.strip().lower()
            val = val.strip()
            mapping = {
                "名称": "name", "name": "name",
                "描述": "description", "description": "description",
                "规则": "rules", "rule": "rules",
            }
            section = mapping.get(key, "description")
            ctx.kb.update_worldview(section, val)
        else:
            ctx.kb.update_worldview("description", content)
        yield event.plain_result(f"✅ 世界观已更新。")

        # 记录贡献者
        self._record_contributor(ctx, event)

        # 触发世界观整理计数
        await self._maybe_refine_worldview(ctx)

    @novel.command("整理世界观", alias={"refine_worldview"})
    async def cmd_refine_worldview(self, event: AstrMessageEvent):
        """手动触发 AI 整理世界观"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        yield event.plain_result("🔄 正在使用 AI 整理世界观，请稍候...")

        provider = self._get_provider_for("worldview")
        novel = ctx.engine.get_novel() if ctx.engine.is_initialized() else {}
        ideas_data = ctx.ideas.get_approved_ideas()
        recent_ideas = "\n".join([
            f"- {i.get('content', '')}" for i in ideas_data[-10:]
        ]) if ideas_data else ""
        story_progress = novel.get("global_summary", "") if novel else ""

        try:
            refined = await ctx.kb.refine_worldview_with_ai(
                provider,
                recent_ideas=recent_ideas,
                story_progress=story_progress,
            )
            parts = ["✅ 世界观整理完成！"]
            if refined.get("name"):
                parts.append(f"世界名称：{refined['name']}")
            if refined.get("description"):
                parts.append(f"描述：{truncate_text(refined['description'], 200)}")
            if refined.get("rules"):
                parts.append(f"规则数：{len(refined['rules'])}")
            if refined.get("locations"):
                parts.append(f"地点数：{len(refined['locations'])}")
            if refined.get("factions"):
                parts.append(f"势力数：{len(refined['factions'])}")
            notes = refined.get("custom", {}).get("ai_notes", "")
            if notes:
                parts.append(f"📋 AI 备注：{truncate_text(notes, 300)}")
            parts.append("💾 旧版世界观已备份到 worldview_backup.json")
            yield event.plain_result("\n".join(parts))
        except Exception as e:
            yield event.plain_result(f"❌ 世界观整理失败：{e}")

    @novel.command("添加人物", alias={"addchar"})
    async def cmd_add_char(self, event: AstrMessageEvent, text: str = ""):
        """添加角色"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        # 从完整消息中提取，防止框架截断参数
        content = text.strip()
        if not content or len(content.split()) < 1:
            raw_msg = (event.message_str or "").strip()
            for prefix in ["/小说 添加人物 ", "/小说 addchar "]:
                if raw_msg.startswith(prefix):
                    content = raw_msg[len(prefix):].strip()
                    break
        parts = content.split(maxsplit=1)
        name = parts[0] if parts else ""
        if not name:
            yield event.plain_result("用法：/小说 添加人物 <名字> <描述>")
            return
        desc = parts[1] if len(parts) > 1 else "暂无描述"
        # 重复检测
        existing = ctx.kb.get_character(name)
        if existing:
            yield event.plain_result(
                f"⚠️ 角色「{name}」已存在（ID: {existing['id']}）。\n"
                f"如需修改设定请使用：/小说 修改人物 {name} <新内容>"
            )
            return
        char = ctx.kb.add_character(name, desc)
        yield event.plain_result(f"✅ 角色「{name}」已添加！（ID: {char['id']}）")

        # 记录贡献者
        self._record_contributor(ctx, event)

        await self._maybe_refine_worldview(ctx)

    @novel.command("修改人物", alias={"editchar"})
    async def cmd_update_char(self, event: AstrMessageEvent, text: str = ""):
        """修改角色设定"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        # 从完整消息中提取，防止框架截断参数
        content = text.strip()
        if not content or len(content.split()) < 2:
            raw_msg = (event.message_str or "").strip()
            for prefix in ["/小说 修改人物 ", "/小说 editchar "]:
                if raw_msg.startswith(prefix):
                    content = raw_msg[len(prefix):].strip()
                    break
        parts = content.split(maxsplit=1)
        name = parts[0] if parts else ""
        if not name:
            yield event.plain_result("用法：/小说 修改人物 <名字> <新描述>")
            return
        new_desc = parts[1] if len(parts) > 1 else ""
        if not new_desc:
            yield event.plain_result("请提供新的角色描述：/小说 修改人物 <名字> <新描述>")
            return
        ch = ctx.kb.get_character(name)
        if not ch:
            yield event.plain_result(f"未找到角色「{name}」。请先使用 /小说 添加人物 添加。")
            return
        ctx.kb.update_character(ch["id"], {"description": new_desc})
        yield event.plain_result(f"✅ 角色「{name}」的设定已更新！\n📝 新描述：{new_desc}")

        # 记录贡献者
        self._record_contributor(ctx, event)

    @novel.command("人物列表", alias={"charlist"})
    async def cmd_list_chars(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chars = ctx.kb.list_characters()
        if not chars:
            yield event.plain_result("暂无角色。使用 /小说 添加人物 <名字> <描述> 添加。")
            return
        lines = ["📋 角色列表"]
        for c in chars:
            lines.append(f"  🟢 {c['name']}：{truncate_text(c.get('description', ''), 50)}")
        yield event.plain_result("\n".join(lines))

    @novel.command("人物", alias={"char"})
    async def cmd_char_detail(self, event: AstrMessageEvent, name: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not name:
            yield event.plain_result("用法：/小说 人物 <名字>")
            return
        ch = ctx.kb.get_character(name)
        if not ch:
            yield event.plain_result(f"未找到角色「{name}」")
            return
        lines = [
            f"🧑 {ch['name']}",
            f"ID：{ch['id']}",
            f"描述：{ch.get('description', '无')}",
        ]
        if ch.get("background"):
            lines.append(f"背景：{ch['background']}")
        if ch.get("abilities"):
            lines.append(f"能力：{'、'.join(ch['abilities'])}")
        if ch.get("relationships"):
            lines.append(f"关系：{'、'.join(str(r) for r in ch['relationships'])}")
        if ch.get("status"):
            lines.append(f"状态：{ch['status']}")
        yield event.plain_result("\n".join(lines))

    # ====== 风格指令 ======

    @novel.command("风格列表", alias={"styles"})
    async def cmd_list_styles(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        styles = ctx.kb.list_styles()
        if not styles:
            yield event.plain_result("暂无风格。使用 /小说 添加风格 <名称> 创建。")
            return
        novel = ctx.engine.get_novel() if ctx.engine.is_initialized() else {}
        cur = novel.get("current_style", "")
        lines = ["🎨 可用风格"]
        for s in styles:
            n = s["name"]
            cnt = len(s.get("samples", []))
            mark = " ★当前" if n == cur else ""
            lines.append(f"  • {n}{mark}（{cnt}个样本）")
        yield event.plain_result("\n".join(lines))

    @novel.command("添加风格", alias={"addstyle"})
    async def cmd_add_style(self, event: AstrMessageEvent, name: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not name:
            yield event.plain_result("用法：/小说 添加风格 <名称>")
            return
        existing = ctx.kb.get_style(name)
        if existing:
            yield event.plain_result(f"风格「{name}」已存在（{len(existing.get('samples', []))}个样本）")
            return
        ctx.kb.add_style(name)
        gid = event.get_group_id() or ""
        self._pending_style[gid] = name
        yield event.plain_result(
            f"✅ 风格「{name}」已创建！\n"
            f"📝 现在请直接在群里发送该风格的示例文本，每条消息自动收集。\n"
            f"完成后发送 /小说 完成风格 结束收集。"
        )

    @novel.command("完成风格", alias={"finishstyle"})
    async def cmd_finish_style(self, event: AstrMessageEvent):
        gid = event.get_group_id() or ""
        if gid not in self._pending_style:
            yield event.plain_result("当前没有正在收集的风格。")
            return
        style_name = self._pending_style.pop(gid)
        ctx = self._get_ctx(event)
        if ctx:
            style = ctx.kb.get_style(style_name)
            count = len(style.get("samples", [])) if style else 0
            yield event.plain_result(f"✅ 风格「{style_name}」收集完成，共 {count} 条样本。")
        else:
            yield event.plain_result(f"✅ 风格「{style_name}」收集完成。")

    @novel.command("风格样本", alias={"stylesample"})
    async def cmd_style_sample(self, event: AstrMessageEvent, text: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        parts = text.strip().split(maxsplit=1)
        name = parts[0] if parts else ""
        if not name:
            yield event.plain_result("用法：/小说 风格样本 <名称> <文本>")
            return
        text = parts[1] if len(parts) > 1 else ""
        if not text:
            yield event.plain_result("请提供示例文本。")
            return
        ok = ctx.kb.add_style_sample(name, text)
        if ok:
            style = ctx.kb.get_style(name)
            count = len(style.get("samples", [])) if style else 0
            yield event.plain_result(f"📝 已添加为「{name}」样本（第 {count} 条）。")
        else:
            yield event.plain_result(f"风格「{name}」不存在。")

    @novel.command("切换风格", alias={"setstyle"})
    async def cmd_set_style(self, event: AstrMessageEvent, name: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not name:
            yield event.plain_result("用法：/小说 切换风格 <名称>")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("请先初始化小说。")
            return
        ok = ctx.engine.set_style(name)
        if ok:
            yield event.plain_result(f"✅ 已切换到「{name}」风格。")
        else:
            yield event.plain_result(f"❌ 未找到风格「{name}」，请检查名称。")

    # ====== 创意指令 ======

    @novel.command("创意", alias={"idea"})
    async def cmd_idea(self, event: AstrMessageEvent, text: str = ""):
        """提交创意并自动打分+冲突检测"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        content = text.strip()
        if not content:
            yield event.plain_result("用法：/小说 创意 <内容>")
            return

        author = event.get_sender_id() or "unknown"
        # 获取 3 个评分 AI provider
        scoring_providers = [
            self._get_provider_for("idea_scoring_1"),
            self._get_provider_for("idea_scoring_2"),
            self._get_provider_for("idea_scoring_3"),
        ]

        # 记录贡献者
        self._record_contributor(ctx, event)
        novel = ctx.engine.get_novel() if ctx.engine.is_initialized() else {}
        novel_title = novel.get("title", "未命名")
        novel_synopsis = novel.get("synopsis", "")
        threshold = self._cfg_int("score_threshold", 70)
        vote_duration = self._cfg_int("vote_duration_minutes", 30)

        yield event.plain_result("💡 收到创意，正在由 3 个 AI 模型进行评分...")

        try:
            # 阶段1：提交创意
            idea = ctx.ideas.submit_idea(
                author=event.get_sender_name() or author,
                author_id=author,
                content=content,
            )

            # 阶段2：3个AI分别打分
            scored_idea = await ctx.ideas.score_idea(
                idea_id=idea["id"],
                providers=scoring_providers,
                novel_title=novel_title,
                novel_synopsis=novel_synopsis,
            )
            if scored_idea:
                idea = scored_idea

            avg = idea.get("weighted_avg", 0)
            scores = idea.get("scores", [])
            score_lines = []
            for s in scores:
                model = s.get('model_name', '未知模型')
                score_lines.append(
                    f"  {model}: {s.get('score', '?')}分 — {s.get('reason', '')}"
                )

            if avg < threshold:
                ctx.ideas.reject_idea(idea["id"])
                yield event.plain_result(
                    f"❌ 创意评分未通过（均分 {avg:.1f}，阈值 {threshold}）\n"
                    + "\n".join(score_lines)
                    + "\n\n💡 如仍想采纳，可发送 /小说 强制采纳"
                )
                return

            yield event.plain_result(
                f"✅ 评分通过（均分 {avg:.1f}），正在进行冲突检测..."
            )

            # 阶段2：冲突检测
            conflict = await ctx.ideas.check_conflict(
                idea_id=idea["id"],
                provider=scoring_providers[0],
            )

            if conflict and conflict.get("has_conflict"):
                # 发起投票
                vote = ctx.ideas.create_conflict_vote(
                    idea_id=idea["id"],
                    conflict_info=conflict,
                    duration_minutes=vote_duration,
                )
                yield event.plain_result(
                    f"⚠️ 发现冲突！\n"
                    f"冲突详情：{conflict.get('suggestion', '')}\n\n"
                    f"{ctx.votes.format_vote_message(vote)}\n\n"
                    f"投票方式：发送 /小说 投票 <选项字母>"
                )
            else:
                # 无冲突，自动采纳
                ctx.ideas.approve_idea(idea["id"])
                yield event.plain_result(
                    f"✅ 创意已采纳！\n"
                    f"「{truncate_text(content, 80)}」\n"
                    f"均分 {avg:.1f}\n"
                    + "\n".join(score_lines)
                )
                await self._maybe_refine_worldview(ctx)

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 创意处理失败: {e}")
            yield event.plain_result(f"❌ 处理失败：{e}")

    @novel.command("创意列表", alias={"idealist"})
    async def cmd_list_ideas(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        ideas = ctx.ideas.get_approved_ideas()
        if not ideas:
            yield event.plain_result("暂无已采纳的创意。")
            return
        lines = ["💡 已采纳创意"]
        for i in ideas[:20]:
            score_val = i.get('weighted_avg', i.get('avg_score', 0))
            lines.append(f"  • ({score_val:.0f}分) {truncate_text(i.get('content', ''), 60)}")
        yield event.plain_result("\n".join(lines))

    @novel.command("强制创意", alias={"force_idea"})
    async def cmd_force_idea(self, event: AstrMessageEvent, text: str = ""):
        """跳过AI评分直接采纳创意"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        content = text.strip()
        if not content:
            # fallback: 从原始消息提取
            msg = (event.message_str or "").strip()
            for prefix in ["/小说 强制创意 ", "/小说 force_idea "]:
                if msg.startswith(prefix):
                    content = msg[len(prefix):].strip()
                    break
        if not content:
            yield event.plain_result("用法：/小说 强制创意 <内容>")
            return

        author = event.get_sender_id() or "unknown"
        idea = ctx.ideas.submit_idea(
            author=event.get_sender_name() or author,
            author_id=author,
            content=content,
        )
        ctx.ideas.approve_idea(idea["id"])
        self._record_contributor(ctx, event)
        yield event.plain_result(
            f"✅ 创意已强制采纳！（跳过AI评分）\n"
            f"「{truncate_text(content, 80)}」"
        )
        await self._maybe_refine_worldview(ctx)

    @novel.command("强制采纳", alias={"force_approve"})
    async def cmd_force_approve(self, event: AstrMessageEvent):
        """强制通过最近一条被拒绝的创意"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        # 找最近一条被拒的创意
        all_ideas = ctx.ideas.get_all_ideas()
        rejected = [i for i in all_ideas if i.get("status") == "rejected"]
        if not rejected:
            yield event.plain_result("当前没有被拒绝的创意。")
            return
        latest = rejected[-1]
        ctx.ideas.approve_idea(latest["id"])
        self._record_contributor(ctx, event)
        score_val = latest.get('weighted_avg', 0)
        yield event.plain_result(
            f"✅ 创意已强制采纳！\n"
            f"「{truncate_text(latest.get('content', ''), 80)}」\n"
            f"（原评分 {score_val:.1f}）"
        )
        await self._maybe_refine_worldview(ctx)

    # ====== 写作指令 ======

    @novel.command("新章节", alias={"newchapter"})
    async def cmd_new_chapter(self, event: AstrMessageEvent, text: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("请先初始化小说。")
            return
        title = text.strip()
        if not title:
            yield event.plain_result("用法：/小说 新章节 <标题>")
            return
        ch = ctx.engine.add_chapter(title)
        yield event.plain_result(f"✅ 第{ch.get('number', '?')}章「{title}」已创建。")

    @novel.command("写", alias={"write"})
    async def cmd_write(self, event: AstrMessageEvent, text: str = ""):
        """生成新场景"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("请先初始化小说。")
            return
        outline = text.strip()
        if not outline:
            yield event.plain_result("用法：/小说 写 <场景描述>")
            return
        if not ctx.engine.get_current_chapter():
            yield event.plain_result("请先创建章节：/小说 新章节 <标题>")
            return

        max_len = self._cfg_int("max_scene_length", 4000)
        provider = self._get_provider_for("writing")

        # 记录贡献者
        self._record_contributor(ctx, event)

        yield event.plain_result(f"📝 正在生成场景：{truncate_text(outline, 50)}\n请稍候...")

        try:
            scene = await ctx.engine.generate_scene(
                scene_outline=outline,
                provider=provider,
                max_length=max_len,
            )
            content = scene.get("content", "")
            yield event.plain_result(
                f"📖 场景生成完成！\n"
                f"—— {scene.get('title', outline)} ——\n\n"
                f"{truncate_text(content, 2000)}\n\n"
                f"💡 使用 /小说 修正 可进行多 AI 润色修正"
            )
            await self._maybe_refine_worldview(ctx)
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 场景生成失败: {e}")
            yield event.plain_result(f"❌ 场景生成失败：{e}")

    @novel.command("修正", alias={"revise"})
    async def cmd_revise(self, event: AstrMessageEvent):
        """多 AI 修正最新场景"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("请先初始化小说。")
            return

        scene = ctx.engine.get_latest_scene()
        if not scene:
            yield event.plain_result("没有可修正的场景。请先使用 /小说 写 <描述> 生成场景。")
            return

        yield event.plain_result(
            f"✏️ 开始三轮修正：审读 → 修改 → 审校\n"
            f"目标场景：{scene.get('title', '?')} (v{scene.get('version', 1)})\n"
            f"请耐心等待..."
        )

        provider = self._get_provider_for("revision")
        try:
            revised = await ctx.engine.revise_scene(
                scene_id=scene["id"],
                provider=provider,
            )
            content = revised.get("content", "")
            yield event.plain_result(
                f"✅ 修正完成！（v{revised.get('version', '?')}）\n\n"
                f"{truncate_text(content, 2000)}"
            )
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 场景修正失败: {e}")
            yield event.plain_result(f"❌ 修正失败：{e}")

    @novel.command("大纲", alias={"outline"})
    async def cmd_outline(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx or not ctx.engine.is_initialized():
            yield event.plain_result("尚未初始化小说。")
            return
        yield event.plain_result(ctx.engine.get_outline())

    # ====== 投票指令 ======

    @novel.command("投票", alias={"vote"})
    async def cmd_vote(self, event: AstrMessageEvent, option: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        # 先自动关闭过期投票
        expired = ctx.votes.auto_close_expired()
        for ev in expired:
            result_msg = ctx.ideas.apply_vote_result(ev)
            yield event.plain_result(f"⏰ 投票已过期自动关闭！\n{result_msg}")

        if not option:
            vote = ctx.votes.get_latest_active_vote()
            if not vote:
                yield event.plain_result("当前没有进行中的投票。")
                return
            yield event.plain_result(ctx.votes.format_vote_message(vote))
            return

        option = option.upper()
        vote = ctx.votes.get_latest_active_vote()
        if not vote:
            yield event.plain_result("当前没有进行中的投票。")
            return

        user_id = event.get_sender_id() or "unknown"
        ok, msg = ctx.votes.cast_vote(vote["id"], user_id, option)
        yield event.plain_result(msg)

    @novel.command("结束投票", alias={"closevote"})
    async def cmd_close_vote(self, event: AstrMessageEvent):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        vote = ctx.votes.get_latest_active_vote()
        if not vote:
            yield event.plain_result("当前没有进行中的投票。")
            return
        closed = ctx.votes.close_vote(vote["id"])
        if closed:
            result_msg = ctx.ideas.apply_vote_result(closed)
            yield event.plain_result(
                f"{ctx.votes.format_vote_message(closed)}\n\n{result_msg}"
            )
        else:
            yield event.plain_result("关闭投票失败。")

    # ====== 导出指令 ======

    @novel.command("导出", alias={"export"})
    async def cmd_export(self, event: AstrMessageEvent, fmt: str = "txt"):
        """导出小说（支持 txt/epub/pdf）"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("尚未初始化小说。")
            return

        fmt = fmt.lower().strip()
        if fmt not in ("txt", "epub", "pdf"):
            yield event.plain_result("支持的格式：txt / epub / pdf\n用法：/小说 导出 epub")
            return

        novel = ctx.engine.get_novel()
        title = novel.get("title", "小说")
        export_dir = ctx.data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "txt":
            out_path = export_dir / f"{title}.txt"
            export_txt(novel, out_path)
            text = out_path.read_text(encoding="utf-8")
            if len(text) <= 3000:
                yield event.plain_result(text)
            if FileComp is not None:
                try:
                    yield event.chain_result([
                        Plain(f"📄 TXT 导出完成（{len(text)}字）\n"),
                        FileComp(file=str(out_path), name=f"{title}.txt"),
                    ])
                except Exception:
                    yield event.plain_result(f"📄 TXT 导出完成（{len(text)}字）\n文件路径：{out_path}")
            else:
                yield event.plain_result(f"📄 TXT 导出完成（{len(text)}字）\n文件路径：{out_path}")

        elif fmt == "epub":
            out_path = export_dir / f"{title}.epub"
            yield event.plain_result("📚 正在生成 EPUB...")
            result = export_epub(novel, out_path)
            if result:
                if FileComp is not None:
                    try:
                        yield event.chain_result([
                            Plain("✅ EPUB 导出完成！\n"),
                            FileComp(file=str(result), name=f"{title}.epub"),
                        ])
                    except Exception:
                        yield event.plain_result(f"✅ EPUB 导出完成！\n文件路径：{result}")
                else:
                    yield event.plain_result(f"✅ EPUB 导出完成！\n文件路径：{result}")
            else:
                yield event.plain_result(
                    "❌ EPUB 导出失败。\n"
                    "请确认已安装 ebooklib：pip install ebooklib"
                )

        elif fmt == "pdf":
            out_path = export_dir / f"{title}.pdf"
            yield event.plain_result("📄 正在生成 PDF...")
            result = export_pdf(novel, out_path)
            if result:
                if FileComp is not None:
                    try:
                        yield event.chain_result([
                            Plain("✅ PDF 导出完成！\n"),
                            FileComp(file=str(result), name=f"{title}.pdf"),
                        ])
                    except Exception:
                        yield event.plain_result(f"✅ PDF 导出完成！\n文件路径：{result}")
                else:
                    yield event.plain_result(f"✅ PDF 导出完成！\n文件路径：{result}")
            else:
                yield event.plain_result(
                    "❌ PDF 导出失败。\n"
                    "请确认已安装 fpdf2：pip install fpdf2"
                )

    @novel.command("阅读", alias={"read"})
    async def cmd_read(self, event: AstrMessageEvent, chapter_num: str = ""):
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("尚未初始化小说。")
            return
        if not chapter_num:
            ch = ctx.engine.get_current_chapter()
            if not ch:
                yield event.plain_result("暂无章节。")
                return
            chapter_num = str(ch.get("number", 1))
        try:
            num = int(chapter_num)
        except ValueError:
            yield event.plain_result("请输入章节编号（数字）")
            return
        text = ctx.engine.export_chapter(num)
        if not text:
            yield event.plain_result(f"未找到第 {num} 章")
            return
        if len(text) > 3000:
            yield event.plain_result(text[:3000] + f"\n...（共 {len(text)} 字，已截断）")
        else:
            yield event.plain_result(text)

    # ====== 用户介入修正章节 ======

    @novel.command("更改", alias={"revise_chapter"})
    async def cmd_revise_chapter(self, event: AstrMessageEvent, text: str = ""):
        """用户介入修正章节"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.engine.is_initialized():
            yield event.plain_result("请先初始化小说。")
            return

        # 从原始消息中用正则提取章节号和描述（最可靠的方式）
        import re as _re
        msg = (event.message_str or "").strip()
        logger.debug(f"[{PLUGIN_ID}] 更改指令 text={text!r} message_str={msg!r}")

        chapter_num = None
        desc = ""

        # 方式1：正则从原始消息提取 "更改 <数字> <描述>"
        m = _re.search(r'更改\s+(\d+)\s*(.*)', msg)
        if m:
            chapter_num = int(m.group(1))
            desc = m.group(2).strip()
        else:
            # 方式2：从 text 参数解析
            parts = text.strip().split(maxsplit=1)
            if parts:
                try:
                    chapter_num = int(parts[0])
                    desc = parts[1].strip() if len(parts) > 1 else ""
                except ValueError:
                    pass

        if chapter_num is None:
            yield event.plain_result(
                "用法：\n"
                "  /小说 更改 <章节号> <修改描述>\n"
                "  /小说 更改 <章节号> 开始  （进入交互收集模式）"
            )
            return

        ch = ctx.engine.get_chapter_by_number(chapter_num)
        if not ch:
            yield event.plain_result(f"未找到第 {chapter_num} 章。")
            return

        # 模式 B：交互收集模式
        if desc == "开始":
            gid = event.get_group_id() or ""
            self._pending_revision[gid] = {
                "chapter_num": chapter_num,
                "messages": [],
            }
            yield event.plain_result(
                f"📝 已进入第 {chapter_num} 章交互修正模式。\n"
                f"请直接发送你的修改意见（文字/图片描述均可）。\n"
                f"发完后请发送 /小说 结束更改 提交给 AI 修改。"
            )
            return

        # 模式 A：一次性描述
        if not desc:
            yield event.plain_result("请提供修改描述，例如：/小说 更改 1 主角的性格需要更强势一些")
            return

        yield event.plain_result(
            f"✏️ 正在根据你的描述修正第 {chapter_num} 章...\n请稍候..."
        )

        provider = self._get_provider_for("revision")

        # 记录贡献者
        self._record_contributor(ctx, event)
        try:
            result = await ctx.engine.revise_chapter_with_user_input(
                chapter_number=chapter_num,
                user_feedback=desc,
                provider=provider,
            )
            if result:
                yield event.plain_result(
                    f"✅ 第 {chapter_num} 章「{result.get('title', '')}」修正完成！\n"
                    f"使用 /小说 阅读 {chapter_num} 查看修改后的内容。"
                )
            else:
                yield event.plain_result("❌ 修正失败，请稍后重试。")
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 用户介入修正失败: {e}")
            yield event.plain_result(f"❌ 修正失败：{e}")

    @novel.command("结束更改", alias={"finish_revision"})
    async def cmd_finish_revision(self, event: AstrMessageEvent):
        """结束交互修正模式，提交 AI 修改"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        gid = event.get_group_id() or ""
        if gid not in self._pending_revision:
            yield event.plain_result("当前没有正在进行的交互修正。请先使用 /小说 更改 <章节号> 开始")
            return

        pending = self._pending_revision.pop(gid)
        chapter_num = pending["chapter_num"]
        messages = pending["messages"]

        if not messages:
            yield event.plain_result("未收集到任何修改意见，已取消。")
            return

        user_feedback = "\n".join(f"- {m}" for m in messages)

        yield event.plain_result(
            f"✏️ 已收集 {len(messages)} 条修改意见。\n"
            f"正在提交给 AI 修正第 {chapter_num} 章...\n请稍候..."
        )

        provider = self._get_provider_for("revision")

        # 记录贡献者
        self._record_contributor(ctx, event)
        try:
            result = await ctx.engine.revise_chapter_with_user_input(
                chapter_number=chapter_num,
                user_feedback=user_feedback,
                provider=provider,
            )
            if result:
                yield event.plain_result(
                    f"✅ 第 {chapter_num} 章「{result.get('title', '')}」修正完成！\n"
                    f"使用 /小说 阅读 {chapter_num} 查看修改后的内容。"
                )
            else:
                yield event.plain_result("❌ 修正失败，请稍后重试。")
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 交互修正失败: {e}")
            yield event.plain_result(f"❌ 修正失败：{e}")

    # ====== 删除指令 ======

    @novel.command("删除人物", alias={"delchar"})
    async def cmd_delete_char(self, event: AstrMessageEvent, text: str = ""):
        """删除角色"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        name = text.strip()
        if not name:
            yield event.plain_result("用法：/小说 删除人物 <名字>")
            return
        ok, msg = ctx.kb.delete_character(name)
        yield event.plain_result(f"{'✅' if ok else '❌'} {msg}")

    @novel.command("删除设定", alias={"delsetting"})
    async def cmd_delete_setting(self, event: AstrMessageEvent, text: str = ""):
        """删除世界观中的某条设定"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        parts = text.strip().split(maxsplit=1)
        if len(parts) < 2:
            yield event.plain_result(
                "用法：/小说 删除设定 <字段> <关键词>\n"
                "字段可选：rules / locations / history / factions / name / description\n"
                "例如：/小说 删除设定 rules 魔法禁令"
            )
            return
        section, keyword = parts[0], parts[1]
        ok, msg = ctx.kb.delete_worldview_item(section, keyword)
        yield event.plain_result(f"{'✅' if ok else '❌'} {msg}")

    @novel.command("清空世界观", alias={"clearworldview"})
    async def cmd_clear_worldview(self, event: AstrMessageEvent):
        """清空整个世界观"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        ctx.kb.clear_worldview()
        yield event.plain_result("✅ 世界观已清空。可以使用 /小说 设定 重新添加。")

    # ------------------------------------------------------------------
    # 群消息监听：捕获风格样本 + 修正信息收集
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        msg = (event.message_str or "").strip()
        if not msg or msg.startswith("/"):
            return

        gid = event.get_group_id() or ""

        # 检查群白名单
        if not self._allow(event):
            return

        # 检查是否有正在收集的用户介入修正
        if gid in self._pending_revision:
            self._pending_revision[gid]["messages"].append(msg)
            count = len(self._pending_revision[gid]["messages"])
            yield event.plain_result(
                f"📝 已收集第 {count} 条修改意见。"
                f"继续发送或 /小说 结束更改 提交。"
            )
            return

        # 检查是否有正在添加的风格
        if gid in self._pending_style:
            ctx = self._get_group_ctx(gid)
            style_name = self._pending_style[gid]
            ok = ctx.kb.add_style_sample(style_name, msg)
            if ok:
                style = ctx.kb.get_style(style_name)
                count = len(style.get("samples", [])) if style else 0
                yield event.plain_result(
                    f"📝 已添加为「{style_name}」样本（第 {count} 条）。"
                    f"继续发送或 /小说 完成风格 结束。"
                )
            return

        # ---- 群聊小说被动消息收集 ----
        ctx = self._get_group_ctx(gid)
        if ctx.chat_novel.is_collecting():
            sender_name = ""
            try:
                sender_name = event.get_sender_name() or ""
            except Exception:
                pass
            sender_id = ""
            try:
                sender_id = str(event.get_sender_id() or "")
            except Exception:
                pass
            if not sender_name:
                sender_name = sender_id or "未知"

            count = ctx.chat_novel.add_message(sender_name, sender_id, msg)
            threshold = self._cfg_int("chat_novel_threshold", 50)

            # 每累积 threshold 条消息时触发判断
            if count >= threshold and count % threshold == 0:
                # 先评估消息质量
                yield event.plain_result(
                    f"📝 群聊小说消息已达 {count} 条，正在评估内容质量..."
                )
                provider = self._get_provider_for("writing")
                try:
                    sufficient, reason = await ctx.chat_novel.evaluate_quality(provider)
                    if not sufficient:
                        yield event.plain_result(
                            f"📊 消息质量评估：有效内容不足（{reason}）\n"
                            f"📡 将继续收集，攒够下一轮 {threshold} 条后再次判断。"
                        )
                        return

                    yield event.plain_result(
                        f"✅ 内容质量充足（{reason}），开始生成新章节，请稍候..."
                    )
                    chapter = await ctx.chat_novel.generate_chapter(provider)
                    if chapter:
                        content_preview = chapter.get("content", "")[:800]
                        yield event.plain_result(
                            f"📖 群聊小说 第{chapter['number']}章「{chapter['title']}」已完成！\n\n"
                            f"{content_preview}\n\n"
                            f"{'...(内容过长已截断)' if len(chapter.get('content', '')) > 800 else ''}\n"
                            f"📚 共 {len(chapter.get('content', ''))} 字\n"
                            f"💾 使用 /群聊小说 阅读 {chapter['number']} 查看全文\n"
                            f"💾 使用 /群聊小说 导出 pdf 可导出全文"
                        )
                    else:
                        yield event.plain_result("⚠️ 群聊小说章节生成失败，请稍后重试。")
                except Exception as e:
                    logger.error(f"[{PLUGIN_ID}] 群聊小说章节生成异常: {e}")
                    yield event.plain_result(f"⚠️ 群聊小说章节生成出错：{e}")

    # ==================================================================
    # 群聊小说 命令组（独立于 /小说）
    # ==================================================================
    CHAT_NOVEL_HELP = """\
📖 【群聊小说】指令一览

▸ /群聊小说 帮助          显示此帮助
▸ /群聊小说 开始构建 <书名> <要求>  开始收集群聊消息并构建小说
▸ /群聊小说 停止          停止收集
▸ /群聊小说 状态          查看进度
▸ /群聊小说 人物列表       查看角色
▸ /群聊小说 人物 <名字>    角色详情
▸ /群聊小说 阅读 [章节号]   阅读章节
▸ /群聊小说 导出 pdf/epub/txt  导出小说
▸ /群聊小说 修改名称 <新书名>  修改小说名称
▸ /群聊小说 删除          删除本群所有小说数据
"""

    @filter.command_group("群聊小说", alias={"chatnovel"})
    def chat_novel_cmd(self):
        pass

    @chat_novel_cmd.command("帮助", alias={"help"})
    async def cn_help(self, event: AstrMessageEvent):
        yield event.plain_result(self.CHAT_NOVEL_HELP)

    @chat_novel_cmd.command("开始构建", alias={"start"})
    async def cn_start(self, event: AstrMessageEvent, text: str = ""):
        """开始收集群聊消息构建小说"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        # 从完整消息中提取书名和要求（优先从原始消息提取，避免框架截断参数）
        content = ""
        raw_msg = (event.message_str or "").strip()
        for prefix in ["/群聊小说 开始构建 ", "/群聊小说 start "]:
            if raw_msg.startswith(prefix):
                content = raw_msg[len(prefix):].strip()
                break
        if not content:
            content = text.strip()
        if not content:
            yield event.plain_result(
                "用法：/群聊小说 开始构建 <书名> <风格/主题要求>\n"
                "例如：/群聊小说 开始构建 仙途纪事 传统玄幻和仙侠风格"
            )
            return
        # 解析：第一个词为书名，其余为要求
        parts = content.split(maxsplit=1)
        title = parts[0]
        requirements = parts[1].strip() if len(parts) > 1 else ""
        if not requirements:
            yield event.plain_result(
                "请同时提供书名和风格要求。\n"
                "用法：/群聊小说 开始构建 <书名> <风格/主题要求>\n"
                "例如：/群聊小说 开始构建 仙途纪事 传统玄幻和仙侠风格"
            )
            return
        if ctx.chat_novel.is_collecting():
            yield event.plain_result(
                "⚠️ 群聊小说已在收集中。如需重新开始，请先 /群聊小说 停止"
            )
            return
        ctx.chat_novel.start(requirements=requirements, title=title)
        threshold = self._cfg_int("chat_novel_threshold", 50)
        yield event.plain_result(
            f"✅ 群聊小说《{title}》开始构建！\n"
            f"📡 正在收集群聊消息...\n"
            f"🎯 风格要求：{requirements}\n"
            f"📊 每 {threshold} 条消息自动生成一章\n\n"
            f"群友们正常聊天即可，AI 会将聊天内容转化为小说情节。\n"
            f"发送 /群聊小说 停止 可随时停止收集。"
        )

    @chat_novel_cmd.command("停止", alias={"stop"})
    async def cn_stop(self, event: AstrMessageEvent):
        """停止收集群聊消息"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.chat_novel.is_collecting():
            yield event.plain_result("⚠️ 群聊小说当前未在收集中。")
            return
        pending = ctx.chat_novel.get_pending_count()
        ctx.chat_novel.stop()
        chapters = ctx.chat_novel.get_chapter_count()
        yield event.plain_result(
            f"⏹ 群聊小说已停止收集。\n"
            f"📖 共生成 {chapters} 章\n"
            f"📝 {pending} 条未处理消息已保留\n"
            f"💾 使用 /群聊小说 导出 pdf 可导出已有内容"
        )

    @chat_novel_cmd.command("状态", alias={"status"})
    async def cn_status(self, event: AstrMessageEvent):
        """查看群聊小说状态"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        yield event.plain_result(ctx.chat_novel.get_status())

    @chat_novel_cmd.command("人物列表", alias={"charlist"})
    async def cn_charlist(self, event: AstrMessageEvent):
        """查看群聊小说人物列表"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chars = ctx.chat_novel.list_characters()
        if not chars:
            yield event.plain_result("暂无角色。开始收集群聊消息后，AI 会自动创建角色。")
            return
        lines = ["📋 群聊小说 — 人物列表", ""]
        for c in chars:
            real = c.get("real_name", "?")
            novel = c.get("novel_name", "?")
            desc = truncate_text(c.get("description", ""), 40)
            lines.append(f"• {novel}（原型：{real}）— {desc}")
        yield event.plain_result("\n".join(lines))

    @chat_novel_cmd.command("人物", alias={"char"})
    async def cn_char_detail(self, event: AstrMessageEvent, text: str = ""):
        """查看群聊小说角色详情"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        name = text.strip()
        if not name:
            yield event.plain_result("用法：/群聊小说 人物 <名字>")
            return
        ch = ctx.chat_novel.get_character(name)
        if not ch:
            yield event.plain_result(f"未找到角色「{name}」。使用 /群聊小说 人物列表 查看所有角色。")
            return
        lines = [
            f"👤 {ch.get('novel_name', '?')}",
            f"  群昵称：{ch.get('real_name', '?')}",
            f"  描述：{ch.get('description', '暂无')}",
        ]
        yield event.plain_result("\n".join(lines))

    @chat_novel_cmd.command("阅读", alias={"read"})
    async def cn_read(self, event: AstrMessageEvent, text: str = ""):
        """阅读群聊小说章节"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chapters = ctx.chat_novel.get_chapters()
        if not chapters:
            yield event.plain_result("暂无章节。请先使用 /群聊小说 开始构建 开始收集。")
            return
        num_str = text.strip()
        if num_str:
            try:
                num = int(num_str)
            except ValueError:
                yield event.plain_result("用法：/群聊小说 阅读 [章节号]\n不填则阅读最新章节。")
                return
        else:
            num = chapters[-1].get("number", 1)
        ch = ctx.chat_novel.get_chapter_by_number(num)
        if not ch:
            yield event.plain_result(
                f"未找到第 {num} 章。当前共 {len(chapters)} 章。"
            )
            return
        content = ch.get("content", "")
        header = f"📖 第{ch['number']}章「{ch.get('title', '')}」\n{'=' * 30}\n\n"
        # 分段发送避免消息过长
        if len(content) > 2000:
            yield event.plain_result(header + content[:2000] + "\n\n...（续）")
            remaining = content[2000:]
            while remaining:
                chunk = remaining[:2000]
                remaining = remaining[2000:]
                suffix = "\n\n...（续）" if remaining else "\n\n— 本章完 —"
                yield event.plain_result(chunk + suffix)
        else:
            yield event.plain_result(header + content + "\n\n— 本章完 —")

    @chat_novel_cmd.command("修改名称", alias={"rename"})
    async def cn_rename(self, event: AstrMessageEvent, text: str = ""):
        """修改群聊小说名称"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        new_title = text.strip()
        if not new_title:
            yield event.plain_result("用法：/群聊小说 修改名称 <新书名>")
            return
        novel = ctx.chat_novel._load_novel()
        old_title = novel.get("title", "群聊物语")
        novel["title"] = new_title
        ctx.chat_novel._save_novel(novel)
        yield event.plain_result(f"✅ 小说名称已修改：《{old_title}》 → 《{new_title}》")

    @chat_novel_cmd.command("删除", alias={"delete", "reset"})
    async def cn_delete(self, event: AstrMessageEvent):
        """删除当前群聊的所有小说数据"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chapters = ctx.chat_novel.get_chapter_count()
        chars = len(ctx.chat_novel.list_characters())
        ctx.chat_novel.reset()
        yield event.plain_result(
            f"✅ 群聊小说数据已全部删除！\n"
            f"📖 已清除 {chapters} 章内容\n"
            f"👤 已清除 {chars} 个人物\n"
            f"📝 消息缓冲已清空\n\n"
            f"如需重新开始，请使用 /群聊小说 开始构建 <要求>"
        )

    @chat_novel_cmd.command("导出", alias={"export"})
    async def cn_export(self, event: AstrMessageEvent, text: str = ""):
        """导出群聊小说"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chapters = ctx.chat_novel.get_chapters()
        if not chapters:
            yield event.plain_result("暂无章节可导出。")
            return

        fmt = (text.strip() or "txt").lower()
        novel_data = ctx.chat_novel.get_novel_data()
        title = novel_data.get("title", "群聊物语")
        export_dir = ctx.data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            if fmt == "txt":
                txt_content = ctx.chat_novel.export_text()
                out = export_dir / f"{title}.txt"
                out.write_text(txt_content, encoding="utf-8")
            elif fmt == "epub":
                yield event.plain_result("📚 正在生成 EPUB...")
                out = export_epub(novel_data, export_dir / f"{title}.epub")
            elif fmt == "pdf":
                yield event.plain_result("📄 正在生成 PDF...")
                out = export_pdf(novel_data, export_dir / f"{title}.pdf")
            else:
                yield event.plain_result(f"不支持的格式：{fmt}。可选：txt / epub / pdf")
                return

            yield event.plain_result(f"✅ 群聊小说导出完成！")
            # 发送文件到群聊
            if FileComp and out and Path(out).exists():
                try:
                    yield event.chain_result([FileComp(name=Path(out).name, url=f"file://{out}")])
                except Exception as e:
                    logger.warning(f"[{PLUGIN_ID}] 群聊小说文件发送失败: {e}")
                    yield event.plain_result(f"📁 文件路径：{out}")
            else:
                yield event.plain_result(f"📁 文件路径：{out}")

        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 群聊小说导出失败: {e}")
            yield event.plain_result(f"❌ 导出失败：{e}")
