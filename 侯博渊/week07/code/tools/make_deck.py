# -*- coding: utf-8 -*-
"""
生成《换数据集重做抽取对比》答辩 PPT（python-pptx，16:9）。

输入：
  code/outputs/report/summary_main.json     各方法指标（eval/compare_duee10.py 产出）
  code/outputs/report/fin_reference.json    DuEE-Fin 课程数字（对照）
  code/outputs/figures/*.png                图表
输出：
  答辩PPT_DuEE10.pptx（默认放在 hw_duee10 根目录）

用法（用带 python-pptx 的解释器）：
  python tools/make_deck.py --out ../答辩PPT_DuEE10.pptx
"""

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "report"
FIG = ROOT / "outputs" / "figures"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x6D, 0xA4)
ACCENT = RGBColor(0xE4, 0x57, 0x56)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def textbox(slide, l, t, w, h, text, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
            font="微软雅黑", anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.text_frame.word_wrap = True
    box.text_frame.vertical_anchor = anchor
    p = box.text_frame.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return box


def title_bar(slide, title, kicker=""):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    if kicker:
        textbox(slide, 0.6, 0.35, 12, 0.4, kicker, size=12, color=BLUE, bold=True)
        textbox(slide, 0.6, 0.68, 12, 0.7, title, size=27, bold=True)
    else:
        textbox(slide, 0.6, 0.45, 12, 0.7, title, size=27, bold=True)


def footer(slide, n):
    textbox(slide, 0.6, 7.05, 8, 0.3, "第 7 周 · 信息抽取 ｜ 换数据集作业：DuEE 1.0", size=9, color=GRAY)
    textbox(slide, 12.2, 7.05, 0.8, 0.3, str(n), size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def table(slide, rows, l, t, w, h, font_size=10.5, col_widths=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(l), Inches(t), Inches(w), Inches(h))
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.name = "微软雅黑"
                    r.font.bold = (i == 0)
                    r.font.color.rgb = WHITE if i == 0 else RGBColor(0x22, 0x22, 0x22)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
    return tbl


def bullets(slide, items, l=0.8, t=1.7, w=11.8, size=17, color=RGBColor(0x22, 0x22, 0x22)):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(5))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (text, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "– ") + text
        r.font.size = Pt(size if lvl == 0 else size - 2)
        r.font.color.rgb = color if lvl == 0 else GRAY
        r.font.name = "微软雅黑"
        r.font.bold = lvl == 0


def pic(slide, path, l, t, w):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), Inches(l), Inches(t), width=Inches(w))


def build(out_path):
    summary = json.loads((REPORT / "summary_main.json").read_text(encoding="utf-8"))
    fin = json.loads((REPORT / "fin_reference.json").read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in summary["methods"]}
    sub = summary.get("subset", {})

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 封面
    s = prs.slides.add_slide(blank)
    add_bg(s, NAVY)
    textbox(s, 0.9, 2.0, 11.5, 0.5, "第 7 周 · 信息抽取 · 课后作业", size=18, color=RGBColor(0xBF, 0xD3, 0xE6))
    textbox(s, 0.9, 2.6, 11.8, 1.2, "换一份数据集，重做一次抽取方法对比", size=40, bold=True, color=WHITE)
    textbox(s, 0.9, 3.9, 11.8, 0.8, "DuEE 1.0：65 类开放域事件（主实验取前 15 类）· 规则 / BiLSTM-CRF / BERT / 本地大模型",
            size=18, color=RGBColor(0xD6, 0xE4, 0xF0))
    textbox(s, 0.9, 6.3, 11.5, 0.4, "2026-11-25 ｜ 八斗学院 AI 大模型课程", size=12, color=RGBColor(0xBF, 0xD3, 0xE6))

    # 2 作业要求与做法
    s = prs.slides.add_slide(blank)
    title_bar(s, "作业要求与本作业的落点", "WHY / WHAT")
    bullets(s, [
        ("要求：不同模型 / 不同 API / 不同数据集做抽取类任务，对比方法效果差异", 0),
        ("本作业的选择：把课程项目整体换到 DuEE 1.0（2020 千言竞赛，65 类、1.7 万句）", 0),
        ("保持“同口径”不变：同一数据契约、同一套评测脚本、同一批 dev 样本", 1),
        ("方法轴：规则 → BiLSTM-CRF → BERT+Linear → 本地 LLM zero/few-shot（CRF 硬约束因 CPU 预算未跑完）", 1),
        ("数据轴：DuEE-Fin（13 类金融公告）↔ DuEE 1.0（65 类开放域新闻）", 1),
        ("结论先行：换数据集后，规则/小模型的相对排序稳定；但“长句+负样本”的设计前提失效，", 0),
        ("方法优劣依赖数据形状，不是绝对排名", 1),
    ], t=1.6, size=17)

    # 3 数据画像
    s = prs.slides.add_slide(blank)
    title_bar(s, "新数据集长什么样：短句、多事件、65 类长尾", "DATA")
    pic(s, FIG / "fig_dataset_profile.png", 0.5, 1.5, 12.3)
    textbox(s, 0.6, 6.15, 12.2, 0.8,
            "DuEE-Fin：5747 句 / 13 类 / 中位 161 字 / 77.9% 无事件句　→　DuEE 1.0：13860 事件 / 65 类 / 中位 35 字 / 全为事件句（13.1% 句子多事件）",
            size=11, color=GRAY)
    footer(s, 3)

    # 4 实验设计
    s = prs.slides.add_slide(blank)
    title_bar(s, "实验设计：一份数据契约，五条完整路线（CRF 待补）", "DESIGN")
    rows_t = [
        ["环节", "做法", "为什么"],
        ["数据", "单事件句 6394/793（与课程口径一致）；多事件句单独留存", "评测脚本按一句一事件设计"],
        ["主实验", "按样本量取前 15 类（覆盖 62% 单事件训练句）", "CPU 预算；标签空间 89，与课程 149 同量级"],
        ["判别式", "BERT-tiny 8.8M（4 层/256 隐层）/ 3 epoch / CPU", "BERT-base 4 层在 CPU 上 1 epoch 要 1 小时"],
        ["生成式", "本地 Qwen3-0.6B zero-shot / few-shot（n=60，seed=42）", "无 API Key；同批子集可比"],
        ["评测", "trigger / argument / event 三级 F1 + type_macro + 非法 BIO", "所有方法同一个 evaluate_events.py"],
    ]
    table(s, rows_t, 0.7, 1.55, 11.9, 3.3, font_size=11.5)
    textbox(s, 0.7, 5.2, 11.9, 1.6,
            "⚠ 可比性边界：判别式路线是“8.8M / 3 epoch”的 CPU 压缩配置，绝对值不能和课程 12 层 GPU 数字比；\n"
            "本作业比较的是同一数据、同一评测口径下方法之间的相对差异，以及跨数据集的趋势变化。",
            size=12.5, color=ACCENT)
    footer(s, 4)

    # 5 主结果表
    s = prs.slides.add_slide(blank)
    title_bar(s, "主结果：三级 F1（DuEE 1.0，15 类，dev 793）", "RESULT")
    order = ["rule", "bilstm_crf", "berttiny_linear", "berttiny_crf", "berttiny_crf_hard", "llm_zero", "llm_few"]
    rows_t = [["方法", "Trigger F1", "Argument F1", "Event F1", "type_macro", "训练/推理成本"]]
    for k in order:
        if k not in rows:
            continue
        r = rows[k]
        cost = f'{r["train_s"]}s / {r["ms"]} ms' if r["kind"] != "llm" else f'0 / {r["ms"]} s'
        rows_t.append([r["name"], f'{r["trigger"]:.4f}', f'{r["argument"]:.4f}',
                       f'{r["event"]:.4f}', f'{r["type_macro"]:.4f}', cost])
    table(s, rows_t, 0.55, 1.5, 12.2, 3.9, font_size=10.5,
          col_widths=[3.6, 1.5, 1.6, 1.3, 1.4, 2.6])
    textbox(s, 0.6, 5.75, 12.2, 1.2,
            "读表：① 主指标看 argument F1；② 本地 0.6B 的 JSON 合法率必须一起看（解析失败=0 分）；"
            "③ 规则 type_macro 很高但论元填不满，和课程结论一致。", size=11.5, color=GRAY)
    footer(s, 5)

    # 6 图：方法对比
    s = prs.slides.add_slide(blank)
    title_bar(s, "三级 F1 一图看完：谁强在触发词、谁强在论元", "RESULT")
    pic(s, FIG / "fig_methods.png", 1.0, 1.4, 11.3)
    footer(s, 6)

    # 7 成本与非法
    s = prs.slides.add_slide(blank)
    title_bar(s, "成本-效果与非法 BIO：贵的不一定值", "COST")
    pic(s, FIG / "fig_cost.png", 0.4, 1.5, 6.2)
    pic(s, FIG / "fig_illegal.png", 6.8, 1.5, 6.1)
    footer(s, 7)

    # 8 跨数据集
    s = prs.slides.add_slide(blank)
    title_bar(s, "跨数据集对比：同一方法，换了数据之后", "CROSS-DATASET")
    fm = fin["methods"]
    rows_t = [["方法", "Fin trigger", "Fin arg", "DuEE1.0 trigger", "DuEE1.0 arg", "变化"],
              ["规则（零训练）", f'{fm["rule"]["trigger_f1"]:.3f}', f'{fm["rule"]["argument_f1"]:.3f}',
               f'{rows["rule"]["trigger"]:.3f}' if "rule" in rows else "-",
               f'{rows["rule"]["argument"]:.3f}' if "rule" in rows else "-",
               "触发词↑、论元↓：自由文本实体变多"],
              ["本地 Qwen3-0.6B zero", f'{fm["llm_local_zero_n60"]["trigger_f1"]:.3f}',
               f'{fm["llm_local_zero_n60"]["argument_f1"]:.3f}',
               f'{rows["llm_zero"]["trigger"]:.3f}' if "llm_zero" in rows else "-",
               f'{rows["llm_zero"]["argument"]:.3f}' if "llm_zero" in rows else "-",
               "65 类 schema 变长 → 小模型更吃力"],
              ["BERT-Linear（Fin 12L / 1.0 8.8M）", f'{fm["bert_linear"]["trigger_f1"]:.3f}',
               f'{fm["bert_linear"]["argument_f1"]:.3f}',
               f'{rows["berttiny_linear"]["trigger"]:.3f}' if "berttiny_linear" in rows else "-",
               f'{rows["berttiny_linear"]["argument"]:.3f}' if "berttiny_linear" in rows else "-",
               "算力不同，只看趋势"],
              ]
    table(s, rows_t, 0.5, 1.5, 12.3, 2.6, font_size=10.5, col_widths=[2.6, 1.5, 1.3, 2.0, 1.7, 4.0])
    textbox(s, 0.6, 4.5, 12.2, 2.2,
            "结论：\n"
            "① 零训练/零样本路线是真正可比的：规则在开放域触发词反而更高（词典覆盖广），但论元更差（大量人名/机构名）。\n"
            "② 判别式路线的绝对值被算力预算压低（4 层 / 1 epoch），不能直接与课程 GPU 12 层数字比。\n"
            "③ 数据形状决定设计：DuEE 1.0 没有负样本句 → 课程“先检测挡 78% 负样本”这一层失去意义；13% 多事件句 → 一句一事件的口径成为结构性上限。",
            size=12.5, color=RGBColor(0x22, 0x22, 0x22))
    footer(s, 8)

    # 9 错误分析
    s = prs.slides.add_slide(blank)
    title_bar(s, "错误分析：换数据集后，错法变了吗？", "ERROR")
    bullets(s, [
        ("规则的错误结构几乎不变：arg_spurious 约 56% —— 不是没找到词，是把实体填进了错误角色", 0),
        ("类型判断仍是强项：规则 type_macro 0.84，和 Fin 的 0.81 同量级", 1),
        ("模型侧的典型错误（badcase 见报告）：", 0),
        ("边界错：把“裁员人数 4000 人”抽成“4000”", 1),
        ("角色串味：地点/主体/对象在同一句里互相填错", 1),
        ("多事件句只抽一个：一句里两个事件，模型按设计只输出一个", 1),
        ("结论：错误的大头仍是“角色定义 + 边界”，不是“模型没学会”", 0),
    ], t=1.6, size=16)
    footer(s, 9)

    # 10 结论
    s = prs.slides.add_slide(blank)
    title_bar(s, "结论与可复现清单", "TAKEAWAY")
    bullets(s, [
        ("换数据集不是换数字：先重读数据形状（句长/负样本/多事件/类型数），再决定方法", 0),
        ("同口径下：规则能判类型、填不满槽位；BERT 的预训练收益主要在论元；（CRF/硬约束为课程结论，本次未复测）", 1),
        ("生成式：本地 0.6B 能跑通格式，但 15 类/45 槽位下指标有限 —— 约束解码/解析器仍是最后一公里", 1),
        ("工程：一条命令复现（data/scripts → rule → bilstm → bert → llm → eval/compare_duee10.py）", 0),
        ("交付：REPORT.md / 作业报告.docx / 本 PPT / 全部预测与指标 JSONL", 1),
    ], t=1.7, size=17)
    footer(s, 10)

    prs.save(str(out_path))
    print("written:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "答辩PPT_DuEE10.pptx"))
    build(Path(ap.parse_args().out))
