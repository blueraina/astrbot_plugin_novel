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

from .utils import PLUGIN_ID, truncate_text, generate_cover_image, render_mermaid_to_image, recognize_image
from .knowledge_base import KnowledgeBase
from .idea_manager import IdeaManager
from .novel_engine import NovelEngine
from .vote_manager import VoteManager
from .exporter import export_txt, export_epub, export_pdf
from .chat_novel import ChatNovelEngine
from .prompts import COVER_IMAGE_PROMPT_TEMPLATE


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

    async def _generate_cover(self, novel: dict, output_path: Path) -> Optional[Path]:
        """为小说生成 AI 封面图片"""
        provider = self._get_provider_for("cover_image")

        # 组装提示词
        user_prompt = self._cfg("cover_image_prompt", "").strip()
        if not user_prompt:
            user_prompt = "一张高品质的小说封面，画风精美细腻，光影层次丰富，具有强烈的故事感与氛围感，色彩和谐统一"

        synopsis = novel.get("synopsis", "")
        if not synopsis:
            synopsis = novel.get("global_summary", "暂无简介")

        title = novel.get("title", "未命名小说")

        final_prompt = COVER_IMAGE_PROMPT_TEMPLATE.format(
            user_prompt=user_prompt,
            title=title,
            synopsis=truncate_text(synopsis, 500),
        )

        # 参考图（图生图模式）
        ref_path = None
        ref_cfg = self._cfg("cover_reference_image", "").strip()
        if ref_cfg:
            ref_path = Path(ref_cfg)
            if not ref_path.exists():
                logger.warning(f"[{PLUGIN_ID}] 参考图不存在：{ref_path}，将使用纯文生图")
                ref_path = None

        logger.info(f"[{PLUGIN_ID}] 正在生成封面图片...")
        timeout = self._cfg_int("cover_image_timeout", 180)
        model = self._cfg("cover_image_model", "").strip()
        cover_provider_id = self._cfg("provider_cover_image", "").strip()
        if not model and "gpt-image" in cover_provider_id.lower():
            model = cover_provider_id.rsplit("/", 1)[-1].strip()
        img_size = self._cfg("cover_image_size", "").strip() or "1024x1536"
        image_api_key = self._cfg("cover_image_api_key", "").strip()
        image_base_url = self._cfg("cover_image_base_url", "").strip()
        image_responses_model = self._cfg("cover_image_responses_model", "").strip()
        try:
            result = await generate_cover_image(
                provider=provider,
                prompt=final_prompt,
                output_path=output_path,
                size=img_size,
                reference_image_path=ref_path,
                timeout=timeout,
                model_name=model,
                api_key=image_api_key,
                base_url=image_base_url,
                responses_model=image_responses_model,
            )
            if result:
                logger.info(f"[{PLUGIN_ID}] 封面图片生成成功：{result}")
            else:
                logger.warning(f"[{PLUGIN_ID}] 封面图片生成返回空，将跳过封面")
            return result
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] 封面生成异常: {e}，将跳过封面")
            return None

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

        # 封面生成（仅 EPUB/PDF）
        cover_path = None
        if fmt in ("epub", "pdf") and self._cfg_bool("enable_cover_image", False):
            cover_path = await self._generate_cover(
                novel, export_dir / "cover.png"
            )

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
            result = export_epub(novel, out_path, cover_path)
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
            result = export_pdf(novel, out_path, cover_path)
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
        if msg.startswith("/"):
            return

        gid = event.get_group_id() or ""

        # 检查群白名单
        if not self._allow(event):
            return

        # 如果消息为空（可能是纯图片消息），检查是否需要处理图片
        if not msg:
            # 仅当群聊小说在收集且开启了图片识别时，才继续处理纯图片消息
            has_images = False
            if self._cfg_bool("chat_novel_image_recognition_enabled", False):
                try:
                    ctx_check = self._get_group_ctx(gid) if gid else None
                    if ctx_check and ctx_check.chat_novel.is_collecting():
                        # 检查消息链是否包含图片
                        if hasattr(event, "message_obj") and event.message_obj:
                            chain = getattr(event.message_obj, "message", []) or []
                            for comp in chain:
                                comp_type_name = type(comp).__name__.lower()
                                if "image" in comp_type_name:
                                    has_images = True
                                    break
                except Exception:
                    pass
            if not has_images:
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

            # --- 提取图片组件并进行识别 ---
            image_descriptions: list[str] = []
            image_recognition_enabled = self._cfg_bool(
                "chat_novel_image_recognition_enabled", False
            )
            if image_recognition_enabled:
                try:
                    message_chain = []
                    if hasattr(event, "message_obj") and event.message_obj:
                        message_chain = getattr(event.message_obj, "message", []) or []

                    image_urls: list[str] = []
                    image_paths: list[str] = []
                    for comp in message_chain:
                        comp_type = getattr(comp, "type", "")
                        # 兼容多种判断方式
                        type_name = ""
                        if isinstance(comp_type, str):
                            type_name = comp_type.lower()
                        elif hasattr(comp_type, "value"):
                            type_name = str(comp_type.value).lower()
                        elif hasattr(comp_type, "name"):
                            type_name = str(comp_type.name).lower()
                        else:
                            type_name = type(comp).__name__.lower()

                        if "image" in type_name or type(comp).__name__ == "Image":
                            # 获取图片 URL
                            url = (
                                getattr(comp, "url", "")
                                or getattr(comp, "file", "")
                                or getattr(comp, "path", "")
                                or ""
                            )
                            if url:
                                if url.startswith("http"):
                                    image_urls.append(url)
                                else:
                                    image_paths.append(url)

                    # 对每张图片进行识别
                    if image_urls or image_paths:
                        vision_provider = self._get_provider_for("image_recognition")
                        from pathlib import Path as _Path

                        for img_url in image_urls:
                            try:
                                desc = await recognize_image(
                                    provider=vision_provider,
                                    image_url=img_url,
                                    timeout=60,
                                )
                                if desc:
                                    image_descriptions.append(desc)
                                    logger.info(
                                        f"[{PLUGIN_ID}] 图片识别完成（URL）："
                                        f"{len(desc)}字"
                                    )
                            except Exception as img_e:
                                logger.warning(
                                    f"[{PLUGIN_ID}] 图片识别失败: {img_e}"
                                )

                        for img_path in image_paths:
                            try:
                                desc = await recognize_image(
                                    provider=vision_provider,
                                    image_path=_Path(img_path),
                                    timeout=60,
                                )
                                if desc:
                                    image_descriptions.append(desc)
                                    logger.info(
                                        f"[{PLUGIN_ID}] 图片识别完成（本地）："
                                        f"{len(desc)}字"
                                    )
                            except Exception as img_e:
                                logger.warning(
                                    f"[{PLUGIN_ID}] 图片识别失败: {img_e}"
                                )
                except Exception as e:
                    logger.warning(
                        f"[{PLUGIN_ID}] 图片提取/识别异常: {e}"
                    )

            # 判断是否是机器人自身发送的消息
            bot_id = ""
            try:
                if hasattr(event, "bot_id"):
                    bot_id = str(event.bot_id)
                elif hasattr(event, "self_id"):
                    bot_id = str(event.self_id)
                elif hasattr(event, "get_self_id"):
                    bot_id = str(event.get_self_id())
            except Exception:
                pass

            collect_bot = self._cfg_bool("chat_novel_collect_bot_messages", False)
            if not collect_bot and sender_id and bot_id and sender_id == bot_id:
                return

            # 如果消息没有文字但有图片识别结果，仍然记录
            if not msg and image_descriptions:
                msg = "[发送了图片]"

            count = ctx.chat_novel.add_message(
                sender_name, sender_id, msg,
                image_descriptions=image_descriptions or None,
            )
            threshold = self._cfg_int("chat_novel_threshold", 50)

            # 每累积 threshold 条消息时触发判断
            trigger_gen = False
            last_try = getattr(ctx.chat_novel, '_last_try_count', 0)
            if count >= threshold:
                if count % threshold == 0:
                    trigger_gen = True
                elif count > threshold and last_try < threshold * (count // threshold):
                    trigger_gen = True

            if trigger_gen:
                ctx.chat_novel._last_try_count = count
                # 先评估消息质量
                yield event.plain_result(
                    f"📝 群聊小说消息已达 {count} 条，正在评估内容质量..."
                )
                provider = self._get_provider_for("writing")
                try:
                    eval_timeout = self._cfg_int("chat_novel_eval_timeout", 30)
                    quality_threshold = self._cfg_int("chat_novel_quality_threshold", 40)
                    sufficient, reason = await ctx.chat_novel.evaluate_quality(
                        provider, timeout=eval_timeout,
                        quality_threshold=quality_threshold,
                    )
                    if not sufficient:
                        # 尝试用小模型过滤无用消息
                        filter_enabled = self._cfg_bool("chat_novel_filter_enabled", False)
                        if filter_enabled:
                            filter_provider = self._get_provider_for("chat_novel_filter")
                            yield event.plain_result(
                                f"📊 消息质量评估：有效内容不足（{reason}）\n"
                                f"🔍 正在使用小模型过滤无用消息..."
                            )
                            try:
                                orig, kept = await ctx.chat_novel.filter_messages(
                                    filter_provider, timeout=eval_timeout
                                )
                                yield event.plain_result(
                                    f"✅ 消息过滤完成：{orig} 条 → {kept} 条有效消息\n"
                                    f"📡 将继续收集，攒够下一轮 {threshold} 条后再次判断。"
                                )
                            except Exception as fe:
                                logger.warning(f"[{PLUGIN_ID}] 消息过滤失败: {fe}")
                                yield event.plain_result(
                                    f"📊 消息质量评估：有效内容不足（{reason}）\n"
                                    f"⚠️ 消息过滤失败，将继续收集。"
                                )
                        else:
                            yield event.plain_result(
                                f"📊 消息质量评估：有效内容不足（{reason}）\n"
                                f"📡 将继续收集，攒够下一轮 {threshold} 条后再次判断。"
                            )
                        return

                    yield event.plain_result(
                        f"✅ 内容质量充足（{reason}），开始生成新章节，请稍候..."
                    )
                    max_words = self._cfg_int("chat_novel_max_word_count", 2000)
                    memory_enabled = self._cfg_bool("chat_novel_memory_enabled", True)
                    memory_top_k = self._cfg_int("chat_novel_memory_top_k", 8)
                    plot_check_enabled = self._cfg_bool("chat_novel_plot_check_enabled", True)
                    # 检查是否有强制结局标记
                    is_force_ending = ctx.chat_novel.get_force_ending()
                    chapter = await ctx.chat_novel.generate_chapter(
                        provider, max_word_count=max_words,
                        force_ending=is_force_ending,
                        memory_enabled=memory_enabled,
                        memory_top_k=memory_top_k,
                        plot_check_enabled=plot_check_enabled,
                    )
                    if chapter:
                        preview_enabled = ctx.chat_novel.get_preview_enabled()
                        preview_limit = self._cfg_int("chat_novel_preview_limit", 800)
                        content = chapter.get("content", "")
                        if preview_enabled and content:
                            if preview_limit > 0:
                                content_preview = content[:preview_limit]
                                truncated = len(content) > preview_limit
                            else:
                                content_preview = content
                                truncated = False
                            yield event.plain_result(
                                f"📖 群聊小说 第{chapter['number']}章「{chapter['title']}」已完成！\n\n"
                                f"{content_preview}\n\n"
                                f"{'...(内容过长已截断)' if truncated else ''}\n"
                                f"📚 共 {len(content)} 字\n"
                                f"💾 使用 /群聊小说 阅读 {chapter['number']} 查看全文\n"
                                f"💾 使用 /群聊小说 导出 pdf 可导出全文"
                            )
                        else:
                            yield event.plain_result(
                                f"📖 群聊小说 第{chapter['number']}章「{chapter['title']}」已完成！\n"
                                f"📚 共 {len(content)} 字\n"
                                f"💾 使用 /群聊小说 阅读 {chapter['number']} 查看全文\n"
                                f"💾 使用 /群聊小说 导出 pdf 可导出全文"
                            )
                    else:
                        yield event.plain_result("⚠️ 群聊小说章节生成失败，请稍后重试。")

                    # 强制结局通知
                    if is_force_ending and chapter:
                        yield event.plain_result(
                            "🎬 群聊小说已强制结局！故事圆满收束。\n"
                            "📡 消息收集已自动停止。\n"
                            "💾 使用 /群聊小说 导出 pdf 可导出完整小说。\n"
                            "🔄 如需开启新故事，请使用 /群聊小说 开始构建"
                        )
                except Exception as e:
                    logger.error(f"[{PLUGIN_ID}] 群聊小说章节生成异常: {e}")
                    yield event.plain_result(f"⚠️ 群聊小说章节生成出错：{e}")

    @filter.after_message_sent()
    async def on_bot_message_sent(self, event: AstrMessageEvent):
        """主动捕获机器人自己发出的消息（如果开启了收集）"""
        gid = event.get_group_id() or ""
        if not gid:
            return

        ctx = self._get_group_ctx(gid)
        if not ctx.chat_novel.is_collecting():
            return

        collect_bot = self._cfg_bool("chat_novel_collect_bot_messages", False)
        if not collect_bot:
            return

        bot_id = "bot"
        try:
            if hasattr(event, "bot_id"):
                bot_id = str(event.bot_id)
            elif hasattr(event, "self_id"):
                bot_id = str(event.self_id)
            elif hasattr(event, "get_self_id"):
                bot_id = str(event.get_self_id())
        except Exception:
            pass

        msg = event.message_str or ""
        if not msg and hasattr(event, "message_obj") and event.message_obj:
            msg = getattr(event.message_obj, "message_str", "") or ""

        msg = msg.strip()
        if not msg:
            return

        ctx.chat_novel.add_message("Eila", bot_id, msg)

    # ==================================================================
    # 群聊小说 命令组（独立于 /小说）
    # ==================================================================
    CHAT_NOVEL_HELP = """\
📖 【群聊小说】指令一览

▸ /群聊小说 帮助          显示此帮助
▸ /群聊小说 开始构建 <书名> <要求>  开始收集群聊消息并构建小说
▸ /群聊小说 停止          停止收集
▸ /群聊小说 继续          继续收集（从停止状态恢复）
▸ /群聊小说 状态          查看进度
▸ /群聊小说 设定 <内容>    添加自定义设定（显示在简介中）
▸ /群聊小说 剧情走向 <内容>  指定下一次生成的剧情走向（生成后自动清空）
▸ /群聊小说 结局          下一次生成强制结局并停止收集
▸ /群聊小说 立即生成       跳过阈值立即生成章节
▸ /群聊小说 重写 <章节号> <说明>  重写指定章节
▸ /群聊小说 人物列表       查看角色
▸ /群聊小说 人物 <名字>    角色详情
▸ /群聊小说 修改角色 <名字> <新描述>  修改角色设定
▸ /群聊小说 锁定角色 <名字>  锁定/解锁角色设定
▸ /群聊小说 关系图        生成角色关系图片
▸ /群聊小说 阅读 [章节号]   阅读章节
▸ /群聊小说 导出 pdf/epub/txt  导出小说
▸ /群聊小说 封面生成 停止   停止每次导出重新生成封面
▸ /群聊小说 封面生成 开始   恢复每次导出重新生成封面
▸ /群聊小说 关闭预览        关闭生成章节后的预览文本
▸ /群聊小说 开启预览        开启生成章节后的预览文本
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
        logger.info(f"[{PLUGIN_ID}] cn_start: text='{text}', raw_msg='{raw_msg}'")
        for prefix in ["/群聊小说 开始构建 ", "/群聊小说 start ",
                       "群聊小说 开始构建 ", "群聊小说 start "]:
            idx = raw_msg.find(prefix)
            if idx >= 0:
                content = raw_msg[idx + len(prefix):].strip()
                break
        if not content:
            content = text.strip()
        logger.info(f"[{PLUGIN_ID}] cn_start: parsed content='{content}'")
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
            f"💾 使用 /群聊小说 导出 pdf 可导出已有内容\n"
            f"▶️ 使用 /群聊小说 继续 可恢复收集"
        )

    @chat_novel_cmd.command("继续", alias={"resume"})
    async def cn_resume(self, event: AstrMessageEvent):
        """从停止状态继续收集群聊消息"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if ctx.chat_novel.is_collecting():
            yield event.plain_result("⚠️ 群聊小说已经在收集中，无需继续。")
            return
        ok = ctx.chat_novel.resume()
        if not ok:
            yield event.plain_result(
                "⚠️ 没有可恢复的群聊小说。\n"
                "请使用 /群聊小说 开始构建 <书名> <要求> 创建新的群聊小说。"
            )
            return
        novel = ctx.chat_novel._load_novel()
        title = novel.get("title", "群聊物语")
        chapters = ctx.chat_novel.get_chapter_count()
        pending = ctx.chat_novel.get_pending_count()
        yield event.plain_result(
            f"▶️ 群聊小说《{title}》继续收集！\n"
            f"📖 已有 {chapters} 章\n"
            f"📝 待处理消息 {pending} 条\n"
            f"群友们继续聊天即可，AI 会自动生成新章节。"
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
        title = ctx.chat_novel._strip_chapter_prefix(ch.get("title", "")) or ch.get("title", "")
        content = ctx.chat_novel._strip_leading_chapter_heading(
            ch.get("content", ""), title
        )
        header = f"📖 第{ch['number']}章「{title}」\n{'=' * 30}\n\n"
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

    @chat_novel_cmd.command("设定", alias={"setting"})
    async def cn_setting(self, event: AstrMessageEvent, text: str = ""):
        """添加自定义设定"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        # 从原始消息提取内容
        content = text.strip()
        raw_msg = (event.message_str or "").strip()
        raw_content = ""
        for prefix in [
            "/群聊小说 设定 ", "/群聊小说 setting ",
            "#群聊小说 设定 ", "#群聊小说 setting ",
            "群聊小说 设定 ", "群聊小说 setting ",
        ]:
            idx = raw_msg.find(prefix)
            if idx >= 0:
                raw_content = raw_msg[idx + len(prefix):].strip()
                break
        # AstrBot 的命令参数在长文本或特殊空白下可能已被截断，优先保留原始消息中更完整的设定。
        if raw_content and len(raw_content) > len(content):
            content = raw_content
        if not content:
            yield event.plain_result(
                "用法：/群聊小说 设定 <内容>\n"
                "添加的设定会显示在导出小说的简介中。"
            )
            return
        count = ctx.chat_novel.add_custom_setting(content)
        yield event.plain_result(
            f"✅ 自定义设定已添加！（当前共 {count} 条设定）\n"
            f"📝 内容：{content}\n"
            f"该设定将显示在导出小说的简介中。"
        )

    @chat_novel_cmd.command("剧情走向", alias={"走向", "下一章", "direction", "plot"})
    async def cn_next_plot_direction(self, event: AstrMessageEvent, text: str = ""):
        """指定下一次生成章节的临时剧情走向"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        content = text.strip()
        raw_msg = (event.message_str or "").strip()
        raw_content = ""
        for prefix in [
            "/群聊小说 剧情走向 ", "/群聊小说 走向 ",
            "/群聊小说 下一章 ",
            "/群聊小说 direction ", "/群聊小说 plot ",
            "#群聊小说 剧情走向 ", "#群聊小说 走向 ",
            "#群聊小说 下一章 ",
            "#群聊小说 direction ", "#群聊小说 plot ",
            "群聊小说 剧情走向 ", "群聊小说 走向 ",
            "群聊小说 下一章 ",
            "群聊小说 direction ", "群聊小说 plot ",
        ]:
            idx = raw_msg.find(prefix)
            if idx >= 0:
                raw_content = raw_msg[idx + len(prefix):].strip()
                break
        if raw_content and len(raw_content) > len(content):
            content = raw_content

        if not content:
            current = ctx.chat_novel.get_next_plot_direction()
            if current:
                yield event.plain_result(
                    "当前已设置下一章剧情走向：\n"
                    f"{current}\n\n"
                    "用法：/群聊小说 剧情走向 <内容>\n"
                    "发送 /群聊小说 剧情走向 清空 可取消。"
                )
            else:
                yield event.plain_result(
                    "用法：/群聊小说 剧情走向 <内容>\n"
                    "例如：/群聊小说 剧情走向 下一章让主角团发现内奸，并在结尾留下反转\n"
                    "该走向只影响下一次章节生成，生成成功后自动清空。"
                )
            return

        if content in {"清空", "取消", "删除", "clear", "reset"}:
            ctx.chat_novel.clear_next_plot_direction()
            yield event.plain_result("✅ 已清空下一章剧情走向。")
            return

        ctx.chat_novel.set_next_plot_direction(content)
        yield event.plain_result(
            "✅ 已设置下一章剧情走向。\n"
            f"🧭 {content}\n"
            "该要求只会影响下一次群聊小说生成，生成成功后自动清空。"
        )

    @chat_novel_cmd.command("结局", alias={"ending"})
    async def cn_ending(self, event: AstrMessageEvent):
        """标记下一次生成强制结局"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        if not ctx.chat_novel.is_collecting():
            yield event.plain_result(
                "⚠️ 群聊小说当前未在收集中。\n"
                "请先使用 /群聊小说 开始构建 开始收集。"
            )
            return
        ctx.chat_novel.set_force_ending(True)
        pending = ctx.chat_novel.get_pending_count()
        threshold = self._cfg_int("chat_novel_threshold", 50)
        yield event.plain_result(
            "🎬 已标记强制结局！\n"
            f"📝 当前待处理消息 {pending} 条（每 {threshold} 条触发生成）\n"
            f"下一次生成的章节将作为最终章，AI 会为故事写一个完整的结局。\n"
            f"生成完成后将自动停止收集。\n"
            f"群友们继续聊天即可，或等待消息攒够后自动触发。"
        )

    @chat_novel_cmd.command("立即生成", alias={"generate", "now"})
    async def cn_generate_now(self, event: AstrMessageEvent):
        """立即生成章节，不等待消息数达到阈值"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        pending = ctx.chat_novel.get_pending_count()
        if pending == 0:
            yield event.plain_result(
                "⚠️ 当前没有待处理的消息。\n"
                "请先让群友们聊天积累素材。"
            )
            return
        if not ctx.chat_novel._load_novel().get("title"):
            yield event.plain_result(
                "⚠️ 请先使用 /群聊小说 开始构建 创建小说。"
            )
            return

        yield event.plain_result(
            f"✅ 当前有 {pending} 条消息，跳过阈值检查，立即生成章节中..."
        )

        provider = self._get_provider_for("writing")
        max_words = self._cfg_int("chat_novel_max_word_count", 2000)
        memory_enabled = self._cfg_bool("chat_novel_memory_enabled", True)
        memory_top_k = self._cfg_int("chat_novel_memory_top_k", 8)
        plot_check_enabled = self._cfg_bool("chat_novel_plot_check_enabled", True)
        is_force_ending = ctx.chat_novel.get_force_ending()

        try:
            chapter = await ctx.chat_novel.generate_chapter(
                provider, max_word_count=max_words,
                force_ending=is_force_ending,
                memory_enabled=memory_enabled,
                memory_top_k=memory_top_k,
                plot_check_enabled=plot_check_enabled,
            )
            if chapter:
                content = chapter.get("content", "")
                preview_limit = self._cfg_int("chat_novel_preview_limit", 800)
                preview_enabled = ctx.chat_novel.get_preview_enabled()
                if preview_enabled and content:
                    if preview_limit > 0:
                        content_preview = content[:preview_limit]
                        truncated = len(content) > preview_limit
                    else:
                        content_preview = content
                        truncated = False
                    yield event.plain_result(
                        f"📖 群聊小说 第{chapter['number']}章「{chapter['title']}」已完成！\n\n"
                        f"{content_preview}\n\n"
                        f"{'...(内容过长已截断)' if truncated else ''}\n"
                        f"📚 共 {len(content)} 字\n"
                        f"💾 使用 /群聊小说 阅读 {chapter['number']} 查看全文"
                    )
                else:
                    yield event.plain_result(
                        f"📖 群聊小说 第{chapter['number']}章「{chapter['title']}」已完成！\n"
                        f"📚 共 {len(content)} 字\n"
                        f"💾 使用 /群聊小说 阅读 {chapter['number']} 查看全文"
                    )
                if is_force_ending:
                    yield event.plain_result(
                        "🎬 群聊小说已强制结局！故事圆满收束。\n"
                        "💾 使用 /群聊小说 导出 pdf 可导出完整小说。"
                    )
            else:
                yield event.plain_result("⚠️ 章节生成失败，请稍后重试。")
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 立即生成异常: {e}")
            yield event.plain_result(f"⚠️ 生成失败：{e}")

    @chat_novel_cmd.command("重写", alias={"rewrite"})
    async def cn_rewrite(self, event: AstrMessageEvent, text: str = ""):
        """重写指定章节"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chapters = ctx.chat_novel.get_chapters()
        if not chapters:
            yield event.plain_result("暂无章节可重写。")
            return

        parts = text.strip().split(None, 1)
        if not parts:
            yield event.plain_result(
                "用法：/群聊小说 重写 <章节号> [补充说明]\n"
                "示例：/群聊小说 重写 2 请增加更多战斗场景"
            )
            return

        try:
            ch_num = int(parts[0])
        except ValueError:
            yield event.plain_result("章节号必须是数字。")
            return

        instructions = parts[1] if len(parts) > 1 else ""

        # 检查章节是否存在
        found = any(ch.get("number") == ch_num for ch in chapters)
        if not found:
            yield event.plain_result(
                f"未找到第 {ch_num} 章。\n"
                f"当前共有 {len(chapters)} 章，可重写范围：1-{len(chapters)}"
            )
            return

        yield event.plain_result(
            f"✏️ 正在重写第 {ch_num} 章，请稍候..."
        )

        provider = self._get_provider_for("writing")
        max_words = self._cfg_int("chat_novel_max_word_count", 2000)
        try:
            new_ch = await ctx.chat_novel.rewrite_chapter(
                provider, ch_num, instructions, max_words
            )
            if new_ch:
                content = new_ch.get("content", "")
                yield event.plain_result(
                    f"✅ 第{ch_num}章「{new_ch['title']}」重写完成！\n"
                    f"📚 共 {len(content)} 字\n"
                    f"💾 使用 /群聊小说 阅读 {ch_num} 查看新内容"
                )
            else:
                yield event.plain_result("⚠️ 章节重写失败，请稍后重试。")
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 章节重写异常: {e}")
            yield event.plain_result(f"⚠️ 重写失败：{e}")


    @chat_novel_cmd.command("关系图", alias={"relationship", "graph"})
    async def cn_relationship(self, event: AstrMessageEvent):
        """生成角色关系图图片"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        chapters = ctx.chat_novel.get_chapters()
        chars = ctx.chat_novel.list_characters()
        if not chapters or not chars:
            yield event.plain_result("暂无足够数据生成关系图（需要至少 1 章和 1 个角色）。")
            return

        yield event.plain_result("🔗 正在分析角色关系并生成图片，请稍候...")

        provider = self._get_provider_for("relationship_graph")
        relationship_timeout = self._cfg_int("chat_novel_relationship_timeout", 120)
        try:
            result = await ctx.chat_novel.generate_relationship_graph(
                provider,
                timeout=relationship_timeout,
            )
            if not result or "mermaid_code" not in result:
                yield event.plain_result("⚠️ 关系图生成失败，请稍后重试。")
                return

            mermaid_code = result["mermaid_code"]
            description = result.get("description", "")

            # 渲染为图片
            export_dir = ctx.data_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            img_path = export_dir / "relationship_graph.png"

            rendered = await render_mermaid_to_image(mermaid_code, img_path)
            if rendered and rendered.exists():
                # 发送文字描述
                if description:
                    yield event.plain_result(
                        f"🔗 角色关系图\n\n{description}"
                    )
                # 发送图片文件
                if FileComp:
                    try:
                        yield event.chain_result([
                            FileComp(name="relationship_graph.png",
                                     url=f"file://{rendered}")
                        ])
                    except Exception as fe:
                        logger.warning(f"[{PLUGIN_ID}] 关系图发送失败: {fe}")
                        yield event.plain_result(
                            f"📁 图片已保存：{rendered}\n"
                            f"发送失败，请手动查看。"
                        )
                else:
                    yield event.plain_result(f"📁 图片已保存：{rendered}")
            else:
                # 图片渲染失败，输出 Mermaid 源码
                yield event.plain_result(
                    f"⚠️ 图片渲染失败，输出 Mermaid 源码：\n\n"
                    f"```mermaid\n{mermaid_code}\n```\n\n"
                    f"{description}"
                )
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] 关系图异常: {e}")
            yield event.plain_result(f"⚠️ 关系图生成失败：{e}")

    @chat_novel_cmd.command("修改角色", alias={"editchar"})
    async def cn_edit_char(self, event: AstrMessageEvent, text: str = ""):
        """修改角色描述"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        parts = text.strip().split(None, 1)
        if len(parts) < 2:
            yield event.plain_result(
                "用法：/群聊小说 修改角色 <名字> <新描述>\n"
                "名字可以是角色的小说名或群昵称。"
            )
            return

        name, new_desc = parts[0], parts[1]
        updated = ctx.chat_novel.update_character_desc(name, new_desc)
        if updated:
            locked_tag = " 🔒" if updated.get("locked") else ""
            yield event.plain_result(
                f"✅ 角色「{updated.get('novel_name', '?')}」设定已更新！{locked_tag}\n"
                f"👤 群昵称：{updated.get('real_name', '?')}\n"
                f"📝 新描述：{new_desc}"
            )
        else:
            yield event.plain_result(
                f"未找到角色「{name}」。\n"
                f"使用 /群聊小说 人物列表 查看所有角色。"
            )

    @chat_novel_cmd.command("锁定角色", alias={"lockchar"})
    async def cn_lock_char(self, event: AstrMessageEvent, text: str = ""):
        """锁定/解锁角色设定"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        name = text.strip()
        if not name:
            yield event.plain_result(
                "用法：/群聊小说 锁定角色 <名字>\n"
                "锁定后 AI 不会自动修改该角色的设定。\n"
                "再次执行即可解锁。"
            )
            return

        result = ctx.chat_novel.toggle_character_lock(name)
        if result:
            char, is_locked = result
            status = "🔒 已锁定" if is_locked else "🔓 已解锁"
            desc = (
                "AI 将不会自动修改该角色的设定。"
                if is_locked else
                "AI 可以根据情节自动更新该角色。"
            )
            yield event.plain_result(
                f"{status} 角色「{char.get('novel_name', '?')}」\n"
                f"👤 群昵称：{char.get('real_name', '?')}\n"
                f"{desc}"
            )
        else:
            yield event.plain_result(
                f"未找到角色「{name}」。\n"
                f"使用 /群聊小说 人物列表 查看所有角色。"
            )

    @chat_novel_cmd.command("封面生成")
    async def cn_cover_toggle(self, event: AstrMessageEvent, action: str = ""):
        """控制封面自动生成开关"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return

        action = action.strip()
        if action == "停止":
            ctx.chat_novel.set_cover_auto_generate(False)
            yield event.plain_result(
                "🖼️ 封面自动生成已关闭。\n"
                "后续导出将复用已有封面图片（如有）。\n"
                "使用 /群聊小说 封面生成 开始 可重新开启。"
            )
        elif action == "开始":
            ctx.chat_novel.set_cover_auto_generate(True)
            yield event.plain_result(
                "🖼️ 封面自动生成已开启。\n"
                "每次导出 EPUB/PDF 时将重新生成封面。"
            )
        else:
            current = ctx.chat_novel.get_cover_auto_generate()
            status = "✅ 开启" if current else "⏹ 关闭"
            yield event.plain_result(
                f"🖼️ 封面自动生成状态：{status}\n\n"
                f"用法：\n"
                f"▸ /群聊小说 封面生成 停止  — 停止每次自动生成\n"
                f"▸ /群聊小说 封面生成 开始  — 恢复每次自动生成"
            )

    @chat_novel_cmd.command("关闭预览")
    async def cn_preview_off(self, event: AstrMessageEvent):
        """关闭生成章节后的预览文本"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        ctx.chat_novel.set_preview_enabled(False)
        yield event.plain_result(
            "🔇 章节预览已关闭。\n"
            "生成新章节后将仅发送完成通知，不发送正文预览。\n"
            "使用 /群聊小说 开启预览 可重新开启。"
        )

    @chat_novel_cmd.command("开启预览")
    async def cn_preview_on(self, event: AstrMessageEvent):
        """开启生成章节后的预览文本"""
        if not self._allow(event):
            return
        ctx = self._get_ctx(event)
        if not ctx:
            yield event.plain_result("该指令仅允许在群聊使用。")
            return
        ctx.chat_novel.set_preview_enabled(True)
        preview_limit = self._cfg_int("chat_novel_preview_limit", 800)
        yield event.plain_result(
            "🔔 章节预览已开启。\n"
            f"生成新章节后将发送前 {preview_limit} 字的正文预览。\n"
            "使用 /群聊小说 关闭预览 可关闭。"
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

        # 封面生成（仅 EPUB/PDF）
        cover_path = None
        if fmt in ("epub", "pdf") and self._cfg_bool("enable_cover_image", False):
            cover_file = export_dir / "cover.png"
            auto_gen = ctx.chat_novel.get_cover_auto_generate()
            if auto_gen or not cover_file.exists():
                yield event.plain_result("🖼️ 正在生成封面图片...")
                cover_path = await self._generate_cover(
                    novel_data, cover_file
                )
            else:
                # 复用已有封面
                cover_path = cover_file
                logger.info(f"[{PLUGIN_ID}] 复用已有封面：{cover_file}")

        try:
            if fmt == "txt":
                txt_content = ctx.chat_novel.export_text()
                out = export_dir / f"{title}.txt"
                out.write_text(txt_content, encoding="utf-8")
            elif fmt == "epub":
                yield event.plain_result("📚 正在生成 EPUB...")
                out = export_epub(novel_data, export_dir / f"{title}.epub", cover_path)
            elif fmt == "pdf":
                yield event.plain_result("📄 正在生成 PDF...")
                out = export_pdf(novel_data, export_dir / f"{title}.pdf", cover_path)
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
