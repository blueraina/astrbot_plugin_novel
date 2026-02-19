"""
导出模块 — 支持 TXT / EPUB / PDF 格式导出
PDF 优先使用本地 MikTeX (xelatex)，如果不可用则回退到 fpdf2
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .utils import PLUGIN_ID


def export_txt(novel: dict, output_path: Path) -> Path:
    """导出为纯文本"""
    lines = [f"《{novel['title']}》", ""]
    # 出场人物章节
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
    if novel.get("synopsis"):
        lines.append(f"【简介】{novel['synopsis']}")
        lines.append("")
    for ch in novel.get("chapters", []):
        lines.append(f"第{ch.get('number', '?')}章 {ch['title']}")
        lines.append("=" * 40)
        lines.append("")
        for sc in ch.get("scenes", []):
            lines.append(sc.get("content", ""))
            lines.append("")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_epub(novel: dict, output_path: Path) -> Optional[Path]:
    """导出为 EPUB 电子书"""
    try:
        from ebooklib import epub
    except ImportError:
        logger.error(f"[{PLUGIN_ID}] 缺少 ebooklib 依赖，无法导出 EPUB。请安装：pip install ebooklib")
        return None

    book = epub.EpubBook()
    title = novel.get("title", "未命名小说")
    book.set_identifier(f"novel-{title}")
    book.set_title(title)
    book.set_language("zh")
    contributors = novel.get("contributors", [])
    author_text = f"作者: {', '.join(contributors)}" if contributors else "群体协作"
    book.add_author(author_text)

    # 样式
    style = epub.EpubItem(
        uid="style",
        file_name="style/default.css",
        media_type="text/css",
        content="""
        body { font-family: "Noto Serif SC", "Source Han Serif CN", serif; line-height: 1.8; margin: 1em; }
        h1 { text-align: center; margin: 2em 0 1em; }
        h2 { margin: 1.5em 0 0.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
        p { text-indent: 2em; margin: 0.5em 0; }
        .scene-title { text-align: center; color: #666; margin: 1em 0; }
        .synopsis { background: #f5f5f5; padding: 1em; border-radius: 4px; margin: 1em 0; }
        """.encode("utf-8"),
    )
    book.add_item(style)

    chapters_epub = []
    spine = ["nav"]

    # 出场人物页
    characters = novel.get("characters", [])
    if characters:
        chars_page = epub.EpubHtml(title="出场人物", file_name="characters.xhtml", lang="zh")
        chars_html_parts = [f'<h1>出场人物</h1>']
        for char in characters:
            novel_name = char.get("novel_name", "")
            real_name = char.get("real_name", "")
            desc = char.get("description", "")
            if novel_name and real_name:
                chars_html_parts.append(f'<p><strong>{novel_name}</strong>（{real_name}）：{desc}</p>')
            elif novel_name:
                chars_html_parts.append(f'<p><strong>{novel_name}</strong>：{desc}</p>')
        chars_page.content = "\n".join(chars_html_parts)
        chars_page.add_item(style)
        book.add_item(chars_page)
        spine.append(chars_page)
        chapters_epub.append(epub.Link("characters.xhtml", "出场人物", "characters"))

    # 简介页
    if novel.get("synopsis"):
        intro = epub.EpubHtml(title="简介", file_name="intro.xhtml", lang="zh")
        # 按换行符分段，使风格和剧情简介分开显示
        synopsis_html = "".join(
            f'<p>{line.strip()}</p>' if line.strip() else '<p style="margin:0.5em 0;">&nbsp;</p>'
            for line in novel["synopsis"].split("\n")
        )
        intro.content = f'<h1>《{title}》</h1><div class="synopsis">{synopsis_html}</div>'
        intro.add_item(style)
        book.add_item(intro)
        spine.append(intro)
        chapters_epub.append(epub.Link("intro.xhtml", "简介", "intro"))

    # 章节
    for ch in novel.get("chapters", []):
        ch_num = ch.get("number", "?")
        ch_title = ch.get("title", "")
        ch_id = f"chapter_{ch_num}"

        ep_ch = epub.EpubHtml(
            title=f"第{ch_num}章 {ch_title}",
            file_name=f"{ch_id}.xhtml",
            lang="zh",
        )

        html_parts = [f"<h1>第{ch_num}章 {ch_title}</h1>"]
        for sc in ch.get("scenes", []):
            content = sc.get("content", "")
            paragraphs = content.split("\n")
            for p in paragraphs:
                p = p.strip()
                if p:
                    html_parts.append(f"<p>{p}</p>")
        ep_ch.content = "\n".join(html_parts)
        ep_ch.add_item(style)
        book.add_item(ep_ch)
        spine.append(ep_ch)
        chapters_epub.append(epub.Link(f"{ch_id}.xhtml", f"第{ch_num}章 {ch_title}", ch_id))

    # 目录和书脊
    book.toc = chapters_epub
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(output_path), book, {})
    logger.info(f"[{PLUGIN_ID}] EPUB 导出完成：{output_path}")
    return output_path


# =====================================================================
# LaTeX 特殊字符转义
# =====================================================================
def _escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符"""
    text = text.replace("\\", "\\textbackslash{}")
    specials = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in specials.items():
        text = text.replace(char, replacement)
    return text


def _md_to_latex(text: str) -> str:
    """将 Markdown 粗体/斜体转为 LaTeX 命令，需在 _escape_latex 之前调用"""
    import re
    # **bold** → \textbf{bold}（先处理双星号）
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # *italic* → \textit{italic}（再处理单星号）
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
    return text


def _escape_latex_with_md(text: str) -> str:
    """先转换 Markdown 格式为 LaTeX 命令，再转义特殊字符（保护已转换的命令）"""
    import re
    # 先提取 Markdown 格式并转为 LaTeX
    converted = _md_to_latex(text)
    # 分离出 LaTeX 命令和普通文本分别处理
    parts = re.split(r'(\\textbf\{[^}]*\}|\\textit\{[^}]*\})', converted)
    result = []
    for part in parts:
        if part.startswith('\\textbf{') or part.startswith('\\textit{'):
            # LaTeX 命令内部只转义内容
            cmd = part[:part.index('{')+1]
            inner = part[part.index('{')+1:-1]
            result.append(cmd + _escape_latex(inner) + '}')
        else:
            result.append(_escape_latex(part))
    return ''.join(result)


def _build_latex_content(novel: dict) -> str:
    """构建 LaTeX 文档内容"""
    title = novel.get("title", "未命名小说")
    contributors = novel.get("contributors", [])
    author_text = f"作者: {', '.join(contributors)}" if contributors else "群体协作"

    tex_lines = [
        r"\documentclass[12pt, a4paper]{ctexart}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{titlesec}",
        r"\usepackage{enumitem}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true, linkcolor=blue, pdfauthor={" + _escape_latex(author_text) + r"}, pdftitle={" + _escape_latex(title) + r"}}",
        "",
        r"\titleformat{\section}{\centering\LARGE\bfseries}{第\thesection 章}{1em}{}",
        r"\titleformat{\subsection}{\large\bfseries\centering}{}{0em}{}",
        "",
        r"\title{\Huge " + _escape_latex(title) + r"}",
        r"\author{\parbox{\textwidth}{\centering " + _escape_latex(author_text) + r"}}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        "",
    ]

    # 出场人物章节
    characters = novel.get("characters", [])
    if characters:
        tex_lines.append(r"\section*{出场人物}")
        tex_lines.append(r"\phantomsection")
        tex_lines.append(r"\addcontentsline{toc}{section}{出场人物}")
        tex_lines.append("")
        for char in characters:
            novel_name = char.get("novel_name", "")
            real_name = char.get("real_name", "")
            desc = char.get("description", "")
            if novel_name and real_name:
                tex_lines.append(r"\textbf{" + _escape_latex(novel_name) + r"}（" + _escape_latex(real_name) + r"）：" + _escape_latex(desc))
            elif novel_name:
                tex_lines.append(r"\textbf{" + _escape_latex(novel_name) + r"}：" + _escape_latex(desc))
            tex_lines.append("")
        tex_lines.append(r"\newpage")
        tex_lines.append("")

    if novel.get("synopsis"):
        tex_lines.append(r"\section*{简介}")
        tex_lines.append(r"\phantomsection")
        tex_lines.append(r"\addcontentsline{toc}{section}{简介}")
        tex_lines.append("")
        # 按换行符分段，使风格和剧情简介分开显示
        for para in novel["synopsis"].split("\n"):
            para = para.strip()
            if para:
                tex_lines.append(_escape_latex(para))
                tex_lines.append("")
            else:
                tex_lines.append(r"\vspace{0.5em}")
                tex_lines.append("")
        tex_lines.append(r"\newpage")
        tex_lines.append("")

    for ch in novel.get("chapters", []):
        ch_title = ch.get("title", "")
        tex_lines.append(r"\section{" + _escape_latex(ch_title) + r"}")
        tex_lines.append("")
        for sc in ch.get("scenes", []):
            content = sc.get("content", "")
            for para in content.split("\n"):
                para = para.strip()
                if para:
                    tex_lines.append(_escape_latex_with_md(para))
                    tex_lines.append("")
        tex_lines.append(r"\newpage")
        tex_lines.append("")

    tex_lines.append(r"\end{document}")
    return "\n".join(tex_lines)


def _try_xelatex(novel: dict, output_path: Path) -> Optional[Path]:
    """
    尝试使用 MikTeX xelatex 编译 PDF。
    返回 Path 表示成功，None 表示失败。
    """
    tex_content = _build_latex_content(novel)

    build_dir = output_path.parent / "_latex_build"
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        tex_path = build_dir / "novel.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        env = os.environ.copy()
        env["MIKTEX_ENABLE_INSTALLER"] = "yes"

        for pass_num in range(2):
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "novel.tex"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(build_dir),
                env=env,
            )
            if result.returncode != 0:
                err_text = (result.stderr or "") + (result.stdout or "")
                if "elevated privileges" in err_text:
                    logger.warning(
                        f"[{PLUGIN_ID}] xelatex 拒绝在管理员权限下运行，"
                        f"将回退到 fpdf2。如需使用 xelatex，请以非管理员身份运行 AstrBot。"
                    )
                    return None
                if pass_num == 1:
                    logger.error(
                        f"[{PLUGIN_ID}] xelatex 编译失败 (pass {pass_num + 1}):\n"
                        f"stdout: {result.stdout[-800:]}\n"
                        f"stderr: {result.stderr[-500:]}"
                    )
                    return None

        pdf_in_build = build_dir / "novel.pdf"
        if pdf_in_build.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(pdf_in_build), str(output_path))
            logger.info(f"[{PLUGIN_ID}] PDF（xelatex）导出完成：{output_path}")
            return output_path
        else:
            logger.error(f"[{PLUGIN_ID}] xelatex 编译未生成 PDF 文件")
            return None

    except FileNotFoundError:
        logger.warning(f"[{PLUGIN_ID}] 未找到 xelatex，将回退到 fpdf2")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"[{PLUGIN_ID}] xelatex 编译超时（120秒）")
        return None
    except Exception as e:
        logger.warning(f"[{PLUGIN_ID}] xelatex 失败（{e}），将回退到 fpdf2")
        return None
    finally:
        try:
            import shutil
            if build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)
        except Exception:
            pass


def _try_fpdf2(novel: dict, output_path: Path) -> Optional[Path]:
    """使用 fpdf2 生成 PDF（不需要外部 LaTeX 环境）"""
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error(
            f"[{PLUGIN_ID}] xelatex 和 fpdf2 均不可用，无法导出 PDF。\n"
            f"方案1：以非管理员身份运行 AstrBot（使 xelatex 可用）\n"
            f"方案2：pip install fpdf2"
        )
        return None

    title = novel.get("title", "未命名小说")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    _setup_chinese_font(pdf)

    # ---- 封面 ----
    pdf.add_page()
    pdf.set_font_size(32)
    pdf.ln(60)
    pdf.cell(0, 20, title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font_size(14)
    pdf.ln(10)
    # 封面书签
    pdf.start_section("封面")

    contributors = novel.get("contributors", [])
    author_text = f"作者: {', '.join(contributors)}" if contributors else "群体协作小说"
    pdf.multi_cell(0, 10, author_text, align="C", new_x="LMARGIN", new_y="NEXT")
    if novel.get("synopsis"):
        pdf.add_page()
        # 如果前面没有换页，简介可能接在封面后。如果想让简介独立一页或有书签：
        # 这里简介一般是在封面上，或者接在后面。用户希望目录有简介。
        # fpdf2 的 start_section 会记录当前 Y 位置为书签跳转点。
        pdf.start_section("简介") 
        pdf.ln(20)
        pdf.set_font_size(11)
        pdf.multi_cell(0, 7, f"【简介】{novel['synopsis']}")

    # ---- 出场人物 ----
    characters = novel.get("characters", [])
    if characters:
        pdf.add_page()
        pdf.start_section("出场人物")
        pdf.set_font_size(22)
        pdf.cell(0, 15, "出场人物", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font_size(11)
        for char in characters:
            novel_name = char.get("novel_name", "")
            real_name = char.get("real_name", "")
            desc = char.get("description", "")
            if novel_name and real_name:
                pdf.multi_cell(0, 7, f"{novel_name}（{real_name}）：{desc}")
            elif novel_name:
                pdf.multi_cell(0, 7, f"{novel_name}：{desc}")
            pdf.ln(2)

    # ---- 正文 ----
    for ch in novel.get("chapters", []):
        pdf.add_page()
    for ch in novel.get("chapters", []):
        pdf.add_page()
        ch_num = ch.get("number", "?")
        ch_title = ch.get("title", "")
        pdf.start_section(f"第{ch_num}章 {ch_title}")
        pdf.set_font_size(22)
        pdf.cell(0, 15, f"第{ch_num}章 {ch_title}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        for sc in ch.get("scenes", []):
            pdf.set_font_size(11)
            content = sc.get("content", "")
            for para in content.split("\n"):
                para = para.strip()
                if para:
                    # 去除 Markdown 粗体/斜体标记
                    para = para.replace("**", "").replace("*", "")
                    pdf.multi_cell(0, 6, f"  {para}")
                    pdf.ln(2)

    pdf.output(str(output_path))
    logger.info(f"[{PLUGIN_ID}] PDF（fpdf2）导出完成：{output_path}")
    return output_path


def _setup_chinese_font(pdf):
    """尝试为 fpdf2 PDF 配置中文字体"""
    import platform

    # 尝试系统字体
    font_paths = []
    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_paths = [
            os.path.join(windir, "Fonts", "msyh.ttc"),     # 微软雅黑
            os.path.join(windir, "Fonts", "simsun.ttc"),    # 宋体
            os.path.join(windir, "Fonts", "simhei.ttf"),    # 黑体
        ]
    elif platform.system() == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdf.add_font("chinese", style="", fname=fp)
                pdf.set_font("chinese", size=11)
                return
            except Exception:
                continue

    # 回退
    pdf.set_font("Helvetica", size=11)
    logger.warning(f"[{PLUGIN_ID}] 未找到中文字体，PDF 可能无法正确显示中文")


# =====================================================================
# PDF 导出入口：xelatex 优先，fpdf2 回退
# =====================================================================
def export_pdf(novel: dict, output_path: Path) -> Optional[Path]:
    """
    导出为 PDF。优先尝试 MikTeX xelatex（更好的排版质量），
    如果 xelatex 不可用或失败则自动回退到 fpdf2。
    """
    # 尝试 xelatex
    result = _try_xelatex(novel, output_path)
    if result:
        return result

    logger.info(f"[{PLUGIN_ID}] xelatex 不可用，回退到 fpdf2 生成 PDF")

    # 回退到 fpdf2
    return _try_fpdf2(novel, output_path)
