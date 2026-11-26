# -*- coding: utf-8 -*-
"""
把 REPORT.md（简单 Markdown 子集）转成带封面/表格/图片的 Word 报告。

支持：# / ## / ### 标题、普通段落、- 列表、1. 有序列表、| 表格 |、![alt](path)、`代码`、
      **加粗**、> 引用。不追求完整 Markdown 语义，够课程报告用。

用法：
  python tools/md_to_docx.py --md REPORT.md --out 作业报告_DuEE10.docx --title "..." --subtitle "..."
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
CODE_FONT = "Consolas"


def set_cjk(run, font=CJK_FONT):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font)


def add_runs(par, text):
    """处理 **加粗** 与 `代码`。"""
    pos = 0
    for m in re.finditer(r"(\*\*.+?\*\*|`.+?`)", text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()])
            set_cjk(r)
        tok = m.group(0)
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2])
            r.bold = True
        else:
            r = par.add_run(tok[1:-1])
            r.font.name = CODE_FONT
            r.font.size = Pt(9.5)
        set_cjk(r)
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:])
        set_cjk(r)


def shade(cell, color="DCE6F1"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def shade_paragraph(par, color="F2F2F2"):
    p_pr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    p_pr.append(shd)


def add_table(doc, rows):
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            add_runs(par, row[j] if j < len(row) else "")
            for r in par.runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
            if i == 0:
                shade(cell)
    doc.add_paragraph()


def build(md_path, out_path, title, subtitle):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.name = CJK_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    style.paragraph_format.line_spacing = 1.25
    style.paragraph_format.space_after = Pt(4)

    # ── 封面 ──
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(26)
    set_cjk(r, CJK_FONT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(subtitle)
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    set_cjk(r, CJK_FONT)
    for _ in range(10):
        doc.add_paragraph()
    for line in ("课程：第 7 周 · 信息抽取", "作业：换数据集重做抽取方法对比",
                 "数据：DuEE 1.0（65 类开放域事件，2020 千言竞赛）", "日期：2026-11-25"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        r.font.size = Pt(11)
        set_cjk(r)
    doc.add_page_break()

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            p = doc.add_heading(level=1)
            add_runs(p, line[2:])
            for r in p.runs:
                r.font.size = Pt(17)
                set_cjk(r, CJK_FONT)
            i += 1
            continue
        if line.startswith("## "):
            p = doc.add_heading(level=2)
            add_runs(p, line[3:])
            for r in p.runs:
                r.font.size = Pt(13.5)
                set_cjk(r, CJK_FONT)
            i += 1
            continue
        if line.startswith("### "):
            p = doc.add_heading(level=3)
            add_runs(p, line[4:])
            for r in p.runs:
                r.font.size = Pt(11.5)
                set_cjk(r, CJK_FONT)
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束围栏
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run("\n".join(code_lines))
            r.font.name = CODE_FONT
            r.font.size = Pt(8.5)
            set_cjk(r, CODE_FONT)
            shade_paragraph(p)
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[-: ]+", "".join(cells)):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue
        if line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if m:
                img = Path(m.group(2))
                if not img.is_absolute():
                    img = (md_path.parent / img).resolve()
                if img.exists():
                    doc.add_picture(str(img), width=Cm(16))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p.add_run(m.group(1))
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
                    set_cjk(r)
            i += 1
            continue
        if line.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            add_runs(p, line.lstrip("> ").strip())
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                r.italic = True
            i += 1
            continue
        if line.strip() in ("---", "***", "___"):
            i += 1
            continue
        if re.match(r"^[-*] ", line):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, line[2:])
            i += 1
            continue
        m_num = re.match(r"^(\d+)\. (.*)$", line)
        if m_num:
            # 不用 Word 的自动编号：多个列表会跨节连续编号（第 2 个列表从 7. 开始），
            # 报告里看起来像“章节串号”。直接把序号写进文本，最稳。
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.45)
            add_runs(p, m_num.group(1) + ". " + m_num.group(2))
            i += 1
            continue
        p = doc.add_paragraph()
        add_runs(p, line)
        i += 1

    doc.save(str(out_path))
    print("written:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="REPORT.md")
    ap.add_argument("--out", default="作业报告_DuEE10.docx")
    ap.add_argument("--title", default="换数据集做抽取对比：DuEE 1.0")
    ap.add_argument("--subtitle", default="规则 / BiLSTM-CRF / BERT / CRF 硬约束 / 本地大模型 六路同口径对比")
    args = ap.parse_args()
    build(Path(args.md), Path(args.out), args.title, args.subtitle)


if __name__ == "__main__":
    main()
