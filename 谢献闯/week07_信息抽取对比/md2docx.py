# -*- coding: utf-8 -*-
"""
Markdown → DOCX（作业报告排版用）

支持：标题(#/##/###)、正文、无序列表、引用、代码块、Markdown 表格、图片、
      行内 **加粗** 与 `代码`。中文字体统一 微软雅黑（东亚字形显式设置 w:eastAsia）。

用法（用 DSH 自带 python，含 python-docx）：
  <dsh-python> w7_ext/md2docx.py --md 实验报告.md --docx 实验报告.docx
"""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CJK_FONT = "微软雅黑"
LATIN_FONT = "Segoe UI"


def set_run_font(run, size=None, bold=None, color=None, mono=False):
    name = "Consolas" if mono else LATIN_FONT
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), "Consolas" if mono else CJK_FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_runs(paragraph, text, size=10.5, bold=False, color=None):
    """把 **bold** 与 `code` 解析成多个 run（支持 **加粗里的 `代码`** 这种嵌套）。"""
    tokens = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            inner = tok[2:-2]
            if "`" in inner:
                add_runs(paragraph, inner, size=size, bold=True, color=color)
            else:
                r = paragraph.add_run(inner)
                set_run_font(r, size=size, bold=True, color=color)
        elif tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1])
            set_run_font(r, size=size - 0.5, bold=bold, color=(0x9C, 0x27, 0x27), mono=True)
        else:
            r = paragraph.add_run(tok)
            set_run_font(r, size=size, bold=bold, color=color)


def shade_cell(cell, hex_color="F2F2F2"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(doc, rows):
    if not rows:
        return
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cell_text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, cell_text, size=9, bold=True)
        shade_cell(cell)
    for row in body:
        cells = table.add_row().cells
        for i in range(len(header)):
            txt = row[i] if i < len(row) else ""
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            add_runs(p, txt, size=9)
    # 表格后留白
    doc.add_paragraph()


def add_image(doc, path, caption=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(15.5))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption)
        set_run_font(r, size=9, color=(0x59, 0x59, 0x59))


BLOCK_START = re.compile(r"^(```|\||!\[|#{1,4}\s|>|[-*]\s|\d+\.\s)")


def take_wrapped(lines, i, first):
    """把一段 Markdown 的软换行续行拼成同一段（中文直接相连，不加空格）。"""
    parts = [first.strip()]
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if not s or BLOCK_START.match(s):
            break
        parts.append(s)
        i += 1
    return "".join(parts), i


def parse_table_block(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # 分隔行
        rows.append(cells)
    return rows


def convert(md_path: Path, docx_path: Path, base_dir: Path):
    # utf-8-sig：PowerShell 5.1 的 -Encoding UTF8 会带 BOM，不剥掉会把 "## 标题" 当正文
    lines = md_path.read_text(encoding="utf-8-sig").splitlines()
    doc = Document()

    # 页面与默认样式
    sec = doc.sections[0]
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # 分隔线：跳过（不把 "---" 当正文）
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            for k, b in enumerate(buf):
                if k:
                    p.add_run().add_break()
                r = p.add_run(b)
                set_run_font(r, size=8.5, mono=True, color=(0x33, 0x33, 0x33))
            continue

        # 表格
        if s.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, parse_table_block(block))
            continue

        # 图片
        m = re.match(r"!\[(.*?)\]\((.+?)\)", s)
        if m:
            caption, src = m.group(1), m.group(2)
            cand = [base_dir / src, Path.cwd() / src]
            p = next((c for c in cand if c.exists()), None)
            if p is not None:
                add_image(doc, p, caption)
            else:
                add_runs(doc.add_paragraph(), f"[缺失图片] {src}", size=9)
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, text, size=18, bold=True)
                p.paragraph_format.space_after = Pt(10)
            else:
                p = doc.add_paragraph()
                size = {2: 14, 3: 12, 4: 11}[level]
                add_runs(p, text, size=size, bold=True,
                         color=(0x1F, 0x3B, 0x63) if level == 2 else (0x33, 0x33, 0x33))
                p.paragraph_format.space_before = Pt(10 if level == 2 else 6)
                p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # 引用
        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            add_runs(p, s.lstrip("> ").strip(), size=9.5, color=(0x59, 0x59, 0x59))
            i += 1
            continue

        # 列表
        m = re.match(r"^([-*]|\d+\.)\s+(.*)$", s)
        if m:
            text, i = take_wrapped(lines, i, m.group(2))
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, text, size=10.5)
            p.paragraph_format.space_after = Pt(2)
            continue

        # 普通段落（把软换行的续行拼回去，避免一句中文被拆成多段）
        text, i = take_wrapped(lines, i, s)
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.25
        add_runs(p, text, size=10.5)

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))
    print(f"saved {docx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--docx", required=True)
    args = ap.parse_args()
    md = Path(args.md).resolve()
    convert(md, Path(args.docx).resolve(), md.parent)


if __name__ == "__main__":
    main()
