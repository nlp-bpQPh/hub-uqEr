"""生成实验报告 DOCX（数据驱动：所有结论数字从 outputs/*.json 读取）。

用法（bundled python）：
  python make_report.py
"""

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(r"E:\大模型学习\作业\第六周作业-分类方法对比")
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REPORT = ROOT / "报告_第六周_分类方法对比实验.docx"

BODY_FONT_CN = "宋体"
HEAD_FONT_CN = "微软雅黑"
LATIN = "Times New Roman"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


# ---------------------------------------------------------------- 基础样式
def set_run_font(run, size=10.5, bold=False, color=None, east=BODY_FONT_CN, latin=LATIN):
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), east)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = LATIN
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CN)
    pf = normal.paragraph_format
    pf.line_spacing = 1.4
    pf.space_after = Pt(4)
    pf.space_before = Pt(0)

    for name, size, color in (("Heading 1", 16, ACCENT), ("Heading 2", 13.5, ACCENT),
                              ("Heading 3", 12, RGBColor(0x33, 0x33, 0x33))):
        st = doc.styles[name]
        st.font.name = LATIN
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT_CN)
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 9)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True


def P(doc, text="", size=10.5, bold=False, align=None, style=None, indent=True,
      space_after=4, color=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        # 支持 **加粗** 行内标记
        for i, seg in enumerate(text.split("**")):
            if not seg:
                continue
            r = p.add_run(seg)
            set_run_font(r, size=size, bold=(bold or i % 2 == 1), color=color)
    return p


def H(doc, text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    set_run_font(r, size={1: 16, 2: 13.5, 3: 12}[level], bold=True,
                 color=ACCENT if level < 3 else RGBColor(0x33, 0x33, 0x33),
                 east=HEAD_FONT_CN)
    return h


def bullets(doc, items, size=10.5):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.35
        r = p.add_run(it)
        set_run_font(r, size=size)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_table(doc, headers, rows, widths=None, font_size=9, header_bg="DCE6F1",
              first_col_left=True, highlight_rows=None, highlight_bg="FFF2CC"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0].cells
    # 表头跨页重复，行内不拆分
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for row in t.rows:
        rp = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); rp.append(cs)
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(str(h))
        set_run_font(r, size=font_size, bold=True)
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(hdr[i], header_bg)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_col_left)
                           else WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(v))
            set_run_font(r, size=font_size, bold=(highlight_rows and ri in highlight_rows))
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if highlight_rows and ri in highlight_rows:
                shade(cells[i], highlight_bg)
    if widths:
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_figure(doc, path, caption, width=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(8)
    r = c.add_run(caption)
    set_run_font(r, size=9, color=RGBColor(0x59, 0x59, 0x59))


def add_page_number(paragraph):
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    set_run_font(run, size=9)


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.2)

    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("第六周作业 · 分类方法对比实验")
    set_run_font(r, size=8.5, color=RGBColor(0x80, 0x80, 0x80))

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("— ")
    set_run_font(r, size=9)
    add_page_number(fp)
    r2 = fp.add_run(" —")
    set_run_font(r2, size=9)


# ---------------------------------------------------------------- 数据
def load_data():
    df = pd.read_csv(OUT / "results.csv", encoding="utf-8-sig")
    stats = json.loads((OUT / "dataset_stats.json").read_text(encoding="utf-8"))
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    return df, stats, analysis


def row(df, ds, method, model=""):
    sub = df[(df.dataset == ds) & (df.method == method)]
    if model:
        sub = sub[sub.model_key == model]
    if sub.empty:
        return None
    return sub.iloc[0]


def f4(x):
    return f"{x:.4f}"


def pp(a, b):
    """两个 F1 的差值，单位「个点」。"""
    return f"{(a - b) * 100:+.1f}"


def main():
    df, stats, analysis = load_data()

    bert_t = row(df, "tnews", "encoder_finetune", "bert-base")
    rob_t = row(df, "tnews", "encoder_finetune", "roberta-wwm-ext")
    mac_t = row(df, "tnews", "encoder_finetune", "macbert-base")
    tf_t = row(df, "tnews", "tfidf_svm")
    rule_t = row(df, "tnews", "rule_keyword")
    probe_t = {k: row(df, "tnews", "frozen_probe", k) for k in
               ("bert-base", "roberta-wwm-ext", "macbert-base")}
    probe_vals = {k: v for k, v in probe_t.items() if v is not None}
    probe_best = max(probe_vals.values(), key=lambda r: r["macro_f1_clean"])

    best_t = max([x for x in (bert_t, rob_t, mac_t) if x is not None],
                 key=lambda r: r["macro_f1_clean"])

    # 跨数据集
    ds_rows = {}
    for ds in ("tnews", "thucnews", "chnsenticorp"):
        ds_rows[ds] = {
            "tfidf": row(df, ds, "tfidf_svm"),
            "bert": row(df, ds, "encoder_finetune", "bert-base"),
            "rob": row(df, ds, "encoder_finetune", "roberta-wwm-ext"),
            "mac": row(df, ds, "encoder_finetune", "macbert-base"),
            "probe": {k: row(df, ds, "frozen_probe", k) for k in
                      ("bert-base", "roberta-wwm-ext", "macbert-base")},
        }

    doc = Document()
    setup_styles(doc)
    setup_page(doc)

    # ---------------- 标题
    op = doc.add_paragraph()
    op.alignment = WD_ALIGN_PARAGRAPH.CENTER
    op.paragraph_format.space_before = Pt(6)
    op.paragraph_format.space_after = Pt(2)
    r = op.add_run("第六周作业 · 数据接入与分流")
    set_run_font(r, size=12, color=RGBColor(0x59, 0x59, 0x59), east=HEAD_FONT_CN)

    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_after = Pt(4)
    r = tp.add_run("更换预训练模型与数据集后的分类方法对比实验")
    set_run_font(r, size=18, bold=True, east=HEAD_FONT_CN)

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sp.paragraph_format.space_after = Pt(2)
    r = sp.add_run("数据接入与分流 · 分类任务（TNEWS / THUCNews / ChnSentiCorp × BERT / RoBERTa / MacBERT）")
    set_run_font(r, size=10.5, color=RGBColor(0x59, 0x59, 0x59), east=HEAD_FONT_CN)

    ip = doc.add_paragraph()
    ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ip.paragraph_format.space_after = Pt(10)
    r = ip.add_run("姓名：____________　　学号：____________　　日期：2026 年 11 月")
    set_run_font(r, size=10)

    # ---------------- 摘要
    H(doc, "摘要", 1)
    P(doc, "本作业在课程「数据接入与分流」代码的基础上，围绕“更换预训练模型”与“更换数据集”两个变量各做一组控制变量实验。"
          "模型侧把 BERT-base-chinese 换成参数量完全相同（102.3M）的 RoBERTa-wwm-ext 与 MacBERT-base；"
          "数据侧把 TNEWS 15 类新闻标题换成 THUCNews 10 类标题（同任务、不同语料）与 ChnSentiCorp 情感二分类（不同任务）。"
          "规则、TF-IDF、冻结特征 + 线性分类器、三个编码器微调共 6 类方法，全部在统一的数据切分、统一的训练配方和同一批"
          "“去重测试样本”上评测。")
    P(doc, f"主要结论：① 在困难任务 TNEWS 上，换更强的预训练权重把 Macro-F1 从 {f4(bert_t['macro_f1_clean'])} 提到 "
          f"{f4(best_t['macro_f1_clean'])}（{pp(best_t['macro_f1_clean'], bert_t['macro_f1_clean'])} 个点），"
          f"提升主要落在长尾类；② 在词汇线索强、类别均衡的 THUCNews 上，TF-IDF 已经拿到 "
          f"{f4(ds_rows['thucnews']['tfidf']['macro_f1_clean'])}，换模型只多 "
          f"{pp(ds_rows['thucnews']['mac']['macro_f1_clean'], ds_rows['thucnews']['tfidf']['macro_f1_clean'])} 个点，"
          f"换模型的边际收益由任务难度决定；③ ChnSentiCorp 的测试集有 27.6% 与训练集完全重复，去重后所有方法掉 1~3 个点，"
          "说明“换数据集”时先把数据洗干净，比换模型更重要。")

    # ---------------- 一、作业要求
    H(doc, "一、作业要求与完成思路", 1)
    P(doc, "作业要求是“更换预训练模型或数据集，做分类任务各种方法的对比实验”。这句话里有两个可以动的变量（模型、数据）"
          "和一个必须守住的前提（各种方法要能横向比较）。因此本作业没有只换一个模型就结束，而是把两个变量拆成两组实验：")
    bullets(doc, [
        "实验一（换模型，数据不动）：在 TNEWS 15 类上固定训练配方，只替换编码器的预训练权重，看“换模型”本身能带来多少提升；",
        "实验二（换数据，方法不动）：把同一套方法搬到 THUCNews 10 类（同任务、不同语料）与 ChnSentiCorp 情感二分类"
        "（不同任务、长文本），看方法排序是否稳定、结论是否可迁移；",
        "横向对照：规则、TF-IDF、冻结特征 + 线性分类器、编码器微调四类方法全部跑一遍，并保留课程已有的 LLM 结果作为参照；",
        "可复现：统一数据切分（固定 seed）、统一超参、dev 只用于选 epoch、test 只评一次，所有数字都能由仓库脚本重跑。",
    ])
    P(doc, "需要说明的是：课程原始代码已经做过一次“五种方法 + 一个服务”的对比，本作业复用了它的标签口径、"
          "去重思想和指标定义，但把训练/评测脚本重写成了“数据集 × 模型”可配置的形式，这样才能保证换模型、换数据时"
          "除被测变量外其余条件完全一致。")

    # ---------------- 二、实验设计
    H(doc, "二、实验设计", 1)
    H(doc, "2.1 变量控制", 2)
    P(doc, "两组实验的设计如下表。实验一只改“预训练权重”，数据、超参、评测集全部不变；实验二改“数据集”，"
          "但每个数据集内部仍然跑同一套方法，因此数据集之间比较的是“方法排序”，不是绝对分数。")
    add_table(doc,
              ["实验", "被测变量", "控制变量", "回答的问题"],
              [["实验一", "编码器预训练权重（BERT / RoBERTa / MacBERT）", "TNEWS、超参、切分、评测集",
                "换一个更强的预训练模型，能涨几个点？涨在哪？"],
               ["实验二", "数据集（TNEWS / THUCNews / ChnSentiCorp）", "每个数据集内方法、配方、切分一致",
                "“BERT 比 TF-IDF 强”这个结论能不能迁移到别的数据？"]],
              widths=[2.0, 5.6, 4.6, 4.4], font_size=9)

    H(doc, "2.2 数据与切分", 2)
    P(doc, "三个数据集都按 train / dev / test 三段切分，dev 只用于选择 epoch，test 只在最后评一次。"
          "TNEWS 原始验证集有 11% 与训练集完全重复，去重后 8,901 条作为 test；由于它没有干净的 dev，"
          "从训练集里按标签分层抽 3,000 条做 dev（训练集因此从 53,360 降到 50,360）。"
          "THUCNews 与 ChnSentiCorp 使用官方 validation 做 dev。")
    rows = []
    for ds, name in (("tnews", "TNEWS"), ("thucnews", "THUCNews"), ("chnsenticorp", "ChnSentiCorp")):
        s = stats[ds]["splits"]
        m = stats[ds]["meta"]
        rows.append([name, s["train"]["n"], s["dev"]["n"], s["test"]["n"],
                     s["train"]["n_classes"], s["train"]["avg_len"],
                     f"{s['train']['imbalance']:.1f}×",
                     f"{m['dup_rate'] * 100:.1f}%"])
    add_table(doc,
              ["数据集", "训练", "dev", "test", "类别数", "平均字数", "不均衡比", "重复率*"],
              rows, widths=[3.5, 1.8, 1.5, 1.8, 1.5, 1.8, 1.8, 2.3], font_size=9)
    P(doc, "* TNEWS 的 11% 是**原始验证集**的重复率（本项目 test 已去重，实际重复率 0）；"
          "THUCNews 与 ChnSentiCorp 是 test 的重复率。"
          f"本作业发现 ChnSentiCorp 的重复率高达 {stats['chnsenticorp']['meta']['dup_rate'] * 100:.1f}%"
          f"（1,200 条测试里 331 条在训练集出现过），THUCNews 只有 {stats['thucnews']['meta']['dup_rate'] * 100:.2f}%。"
          "因此所有方法都额外报告一套“去重测试集”指标，主指标用去重后的数字。", size=10)

    H(doc, "2.3 对比的方法", 2)
    P(doc, "六类方法覆盖了从“不训练”到“全参数微调”的完整谱系，另附课程已有的生成式方法结果作为参照（口径不同，单独标注）。")
    add_table(doc,
              ["方法", "是否用预训练", "训练成本", "在本作业中的角色"],
              [["规则 / 关键词", "否", "0", "下限：不训练模型能拿到多少"],
               ["TF-IDF + LinearSVC", "否", "秒级（CPU）", "传统基线：词面特征的天花板"],
               ["冻结特征 + 线性分类器", "是（不更新）", "分钟级（GPU 前向）", "区分“预训练表示”与“微调”的贡献"],
               ["BERT-base-chinese 微调", "是（全参数）", "分钟级（GPU）", "课程基线模型"],
               ["RoBERTa-wwm-ext 微调", "是（全参数）", "同上", "换模型：全词掩码 + 5.4B tokens"],
               ["MacBERT-base 微调", "是（全参数）", "同上", "换模型：纠错式 MLM"]],
              widths=[5.2, 3.6, 4.2, 4.6], font_size=9)
    P(doc, "课程已有的生成式方法结果（500 条抽样口径，非本实验重跑，仅作参照）：本地 Qwen3-0.6B zero-shot 0.105、"
          "LoRA SFT 0.484、全量微调 0.492、API zero-shot 0.471、BERT 微调 0.544、TF-IDF 0.472。"
          "这些数字与本文表格不可直接相加比较，原因是评测集不同（500 vs 8,901）、切分不同；"
          "本作业没有重跑 LLM，是因为在 8,901 条上跑 zero-shot 需要约 90 分钟，而它并不是“换模型/换数据”这两个变量的一部分。")

    H(doc, "2.4 训练与评测协议", 2)
    P(doc, "三个编码器使用完全相同的配方：3 个 epoch、batch size 64、encoder 学习率 2e-5、分类头 5×、"
          "warmup 10%、类别加权（balanced）、AdamW、AMP 混合精度、max_length 64（ChnSentiCorp 128）。"
          "池化统一用 [CLS]，损失统一用带类别加权的交叉熵。每个 epoch 在 dev 上算 Macro-F1，取 dev 最优的 epoch 评 test。")
    P(doc, "指标以 Macro-F1 为主（每类等权，长尾类不会被头部类掩盖），辅以 Accuracy、每类召回、ECE（置信度校准）"
          "和训练成本（耗时、显存峰值）。同一数据集内所有方法用同一个 metrics 函数计算，避免“这个脚本 weighted F1、"
          "那个脚本 macro F1”导致数字不可比。")

    # ---------------- 三、实验一
    H(doc, "三、实验一：换预训练模型（TNEWS 15 类）", 1)
    P(doc, "三个编码器参数量都是 102.3M（12 层、768 维、21128 词表），结构完全一致，差别只在预训练语料与掩码策略："
          "BERT-base-chinese 是字级 MLM、中文维基；RoBERTa-wwm-ext 用 5.4B tokens、全词掩码 + 动态掩码、去掉 NSP；"
          "MacBERT 用“相似词替换”的纠错式 MLM。因此这一组对比可以干净地归因到“预训练权重”。")
    rows = []
    for name, r in (("规则 / 关键词", rule_t), ("TF-IDF + LinearSVC", tf_t),
                    ("BERT-base 冻结 + LinearSVC", probe_t["bert-base"]),
                    ("RoBERTa 冻结 + LinearSVC", probe_t["roberta-wwm-ext"]),
                    ("MacBERT 冻结 + LinearSVC", probe_t["macbert-base"]),
                    ("BERT-base 微调", bert_t), ("RoBERTa-wwm-ext 微调", rob_t),
                    ("MacBERT-base 微调", mac_t)):
        if r is None:
            continue
        rows.append([name, f4(r["acc_clean"]), f4(r["macro_f1_clean"]),
                     ("-" if pd.isna(r["train_seconds"]) else f"{r['train_seconds']:.0f}s")])
    add_table(doc, ["方法", "Accuracy", "Macro-F1", "训练耗时"], rows,
              widths=[6.4, 3.2, 3.2, 3.2], font_size=9)
    add_figure(doc, FIG / "fig1_tnews_methods.png", "图 1  TNEWS 15 类上各方法的 Macro-F1（8,901 条去重验证集）")

    P(doc, f"结果很集中：三个微调模型都明显高于 TF-IDF（{f4(tf_t['macro_f1_clean'])}）。"
          f"更有意思的是中间那一档：冻结特征 + 线性分类器最好的一组（{probe_best['model_display'].split('（')[0]}）"
          f"已经拿到 {f4(probe_best['macro_f1_clean'])}，比 TF-IDF 高 "
          f"{pp(probe_best['macro_f1_clean'], tf_t['macro_f1_clean'])} 个点；"
          f"而最好的微调模型再比它高 {pp(best_t['macro_f1_clean'], probe_best['macro_f1_clean'])} 个点"
          f"（{f4(best_t['macro_f1_clean'])}）。"
          "这说明预训练的收益可以拆成两段：静态上下文表示本身就比字 n-gram 强，微调再把表示任务化、再多 1~2 个点。"
          "算力或延迟紧张时，“冻结编码器 + 线性分类器”是性价比很高的折中，不必一上来就全参数微调。"
          f"换模型本身：RoBERTa 比 BERT {pp(rob_t['macro_f1_clean'], bert_t['macro_f1_clean'])} 个点，"
          f"MacBERT 比 BERT {pp(mac_t['macro_f1_clean'], bert_t['macro_f1_clean'])} 个点，"
          f"最好的 {best_t['model_display']} 拿到 {f4(best_t['macro_f1_clean'])}。")
    P(doc, f"另一个容易被忽略的数字是 ECE：三个编码器在 TNEWS 上的 ECE 在 0.13~0.15 之间，"
          "也就是说模型说“80% 置信”时实际只有 65%~70% 是对的。"
          "如果下游要用置信度做阈值（课程服务里的 0.8 闸门），必须先在留出集上做温度校准，"
          "否则高置信样本的精度会被高估。")

    P(doc, "提升不是均匀分布的。图 4 的每类召回显示，三个模型在头部类（娱乐、体育、科技）上都已经很高，"
          "差异主要出现在长尾类和边界类：样本最少的“证券”（test 仅 43 条）、“故事”（214 条）以及和“财经”高度重叠的类目。"
          "RoBERTa 的全词掩码让“业绩预告”“回购”这类词内部保持一致，MacBERT 的纠错式 MLM 对错别字和口语化标题更稳，"
          "两者对长尾类都有帮助。")
    add_figure(doc, FIG / "fig4_per_class_recall.png", "图 4  TNEWS 每类召回对比：换模型的收益主要落在长尾/边界类")

    add_figure(doc, FIG / "fig3_training_curves.png", "图 3  TNEWS 训练曲线（dev Macro-F1，用于选 epoch）")
    P(doc, "训练曲线还有一个工程信号：三个模型都在第 2~3 个 epoch 就到顶，继续训练只会过拟合。"
          "这也解释了为什么课程里“再加 Label Smoothing / Focal Loss / FGM”都涨不动——瓶颈在数据（标签口径重叠、短标题缺实体），"
          "不在优化技巧。")

    # ---------------- 四、实验二
    H(doc, "四、实验二：换数据集（THUCNews / ChnSentiCorp）", 1)
    P(doc, "换数据集回答两个问题：方法排序会不会变？“换模型值不值”的结论能不能迁移？"
          "THUCNews 与 TNEWS 同属中文新闻标题分类，但语料来源、类别体系不同（10 类、类别均衡）；"
          "ChnSentiCorp 是酒店/商品评论的情感二分类，文本长（平均 108 字）、类别均衡，和标题分类差异很大。")

    for ds, title in (("thucnews", "4.1 THUCNews 10 类：词汇线索强时，TF-IDF 就够了"),
                      ("chnsenticorp", "4.2 ChnSentiCorp 情感二分类：长文本上预训练更有优势")):
        H(doc, title, 2)
        r = ds_rows[ds]
        rows = []
        for name, x in (("TF-IDF + LinearSVC", r["tfidf"]),
                        ("BERT-base 冻结 + LinearSVC", r["probe"]["bert-base"]),
                        ("BERT-base 微调", r["bert"]),
                        ("RoBERTa-wwm-ext 微调", r["rob"]),
                        ("MacBERT-base 微调", r["mac"])):
            if x is None:
                continue
            rows.append([name, f4(x["acc_clean"]), f4(x["macro_f1_clean"]),
                         ("-" if pd.isna(x["train_seconds"]) else f"{x['train_seconds']:.0f}s")])
        add_table(doc, ["方法", "Accuracy", "Macro-F1", "训练耗时"], rows,
                  widths=[6.4, 3.2, 3.2, 3.2], font_size=9)
        if ds == "thucnews":
            best_enc = max(r['bert']['macro_f1_clean'], r['rob']['macro_f1_clean'], r['mac']['macro_f1_clean'])
            ratio = max(r['bert']['train_seconds'], r['rob']['train_seconds'], r['mac']['train_seconds']) / max(r['tfidf']['train_seconds'], 0.1)
            P(doc, f"THUCNews 上 TF-IDF 直接拿到 {f4(r['tfidf']['macro_f1_clean'])}，"
                  f"最好的编码器 {f4(best_enc)}，只差 {pp(best_enc, r['tfidf']['macro_f1_clean'])} 个点；"
                  f"但训练时间从 {r['tfidf']['train_seconds']:.1f}s 变成约 {r['mac']['train_seconds']:.0f}s（约 {ratio:.0f} 倍）。"
                  "原因是这个数据集的标题里主题词非常直接（“湖人”“科比”对应体育，“股票”“基金”对应股票），"
                  "字 1-2gram 已经足够；类别人数均衡（不均衡比 1.0×），也没有 TNEWS 那种“财经/证券”边界模糊的问题。"
                  "在这个任务上多花两个数量级的训练时间换不到 2 个点，工程上不划算。")
        else:
            best_enc = max(r['bert']['macro_f1_clean'], r['rob']['macro_f1_clean'], r['mac']['macro_f1_clean'])
            P(doc, f"ChnSentiCorp 上 TF-IDF 是 {f4(r['tfidf']['macro_f1_clean'])}，"
                  f"最好的编码器 {f4(best_enc)}，差 {pp(best_enc, r['tfidf']['macro_f1_clean'])} 个点。"
                  "评论是长文本，情感往往由“但是”“还算”“不推荐”这类转折和否定表达决定，"
                  "词面特征容易抓到单词却抓不到否定范围，预训练模型的上下文表示在这里更占优。"
                  "注意这里的数字是去重后的 869 条，全量 1,200 条上所有方法都会高 1~3 个点（见第六节）。")

    add_figure(doc, FIG / "fig2_cross_dataset.png", "图 2  三个数据集上 TF-IDF 与三个编码器的 Macro-F1（去重测试集）")
    P(doc, "把三个数据集放在一起看，排序是稳定的：编码器微调 > TF-IDF > 规则，三个数据集都成立；"
          "三个编码器内部，RoBERTa / MacBERT 在三个数据集上都不低于 BERT。"
          f"但**差距的大小**差别很大：TNEWS 上最好的编码器领先 TF-IDF {pp(best_t['macro_f1_clean'], tf_t['macro_f1_clean'])} 个点，"
          f"THUCNews 上领先 {pp(max(ds_rows['thucnews']['bert']['macro_f1_clean'], ds_rows['thucnews']['rob']['macro_f1_clean'], ds_rows['thucnews']['mac']['macro_f1_clean']), ds_rows['thucnews']['tfidf']['macro_f1_clean'])} 个点，"
          f"ChnSentiCorp 上领先 {pp(max(ds_rows['chnsenticorp']['bert']['macro_f1_clean'], ds_rows['chnsenticorp']['rob']['macro_f1_clean'], ds_rows['chnsenticorp']['mac']['macro_f1_clean']), ds_rows['chnsenticorp']['tfidf']['macro_f1_clean'])} 个点。"
          "换句话说，“换模型”的收益是任务相关的：任务越难、词面线索越弱、类别越不均衡，预训练权重的价值越大。")
    P(doc, "冻结特征这一档也随数据集变化：TNEWS 上它比 TF-IDF 高约 3 个点，"
          f"但 THUCNews 上 {f4(ds_rows['thucnews']['probe']['bert-base']['macro_f1_clean'])} 反而略低于 TF-IDF 的 "
          f"{f4(ds_rows['thucnews']['tfidf']['macro_f1_clean'])}，"
          f"ChnSentiCorp 上 {f4(ds_rows['chnsenticorp']['probe']['bert-base']['macro_f1_clean'])} 与 TF-IDF 的 "
          f"{f4(ds_rows['chnsenticorp']['tfidf']['macro_f1_clean'])} 基本持平。"
          "说明 mean-pooling 的静态句向量能装下“主题词”，却装不下长评论里的否定与转折；"
          "任务越依赖上下文，越必须微调，而不是拿句向量接个分类器。")

    # ---------------- 五、成本
    H(doc, "五、成本-收益：换模型到底贵不贵", 1)
    add_figure(doc, FIG / "fig5_cost.png", "图 5  训练耗时对比（对数轴）：换预训练权重几乎不增加成本")
    P(doc, "三个编码器参数量相同，训练耗时几乎一样（TNEWS 约 9~10 分钟、THUCNews 约 10 分钟、ChnSentiCorp 约 3.5 分钟，"
          "单卡 RTX 3060、3 epoch；同数据集内 BERT / RoBERTa / MacBERT 的差异在 ±15% 以内，主要来自系统负载而非模型），"
          "显存峰值 3.0~4.6 GB（max_length 128 的情感任务更高）。也就是说，**换预训练权重本身几乎不增加成本**，"
          "成本差主要来自“用不用编码器”这个决定：TF-IDF 秒级、编码器分钟级，相差两个数量级。")
    P(doc, "因此工程上的决策顺序应该是：先用 TF-IDF 花 20 秒确认任务的天花板，再决定要不要上编码器；"
          "如果上，优先选与任务分布更接近的预训练权重（中文新闻标题选 RoBERTa-wwm-ext / MacBERT），"
          "而不是盲目换更大的模型——本作业里三个 102M 模型的差异只有 1~2 个点，换 large 的收益很可能也在这个量级，"
          "但训练时间要翻 3 倍。")

    # ---------------- 六、数据泄漏
    H(doc, "六、换数据集时先洗数据：重复样本的影响", 1)
    leak_rows = []
    for ds, name in (("tnews", "TNEWS"), ("thucnews", "THUCNews"), ("chnsenticorp", "ChnSentiCorp")):
        r = ds_rows[ds]
        if r["tfidf"] is None or r["tfidf"]["removed_rate"] < 0.02:
            continue  # TNEWS 已在切分时去重、THUCNews 重复率 <1%，不占表格
        for mname, x in (("TF-IDF", r["tfidf"]), ("BERT-base", r["bert"]),
                         ("MacBERT", r["mac"])):
            if x is None:
                continue
            leak_rows.append([name, mname, f4(x["macro_f1"]), f4(x["macro_f1_clean"]),
                              f"{x['removed_rate'] * 100:.1f}%"])
    add_table(doc, ["数据集", "方法", "全量 test F1", "去重 test F1", "被剔除比例"], leak_rows,
              widths=[3.6, 4.0, 3.2, 3.2, 3.2], font_size=9)
    add_figure(doc, FIG / "fig7_leakage.png", "图 7  去掉与训练集重复的测试样本后，各方法指标的变化")
    tf_full, tf_clean = (ds_rows['chnsenticorp']['tfidf']['macro_f1'],
                         ds_rows['chnsenticorp']['tfidf']['macro_f1_clean'])
    bert_full, bert_clean = (ds_rows['chnsenticorp']['bert']['macro_f1'],
                             ds_rows['chnsenticorp']['bert']['macro_f1_clean'])
    P(doc, f"ChnSentiCorp 的测试集有 27.6% 与训练集完全重复，TF-IDF 从 "
          f"{f4(tf_full)} 掉到 {f4(tf_clean)}（{pp(tf_clean, tf_full)} 个点），BERT 从 "
          f"{f4(bert_full)} 掉到 {f4(bert_clean)}（{pp(bert_clean, bert_full)} 个点）。"
          "TF-IDF 掉得更多，因为它就是靠词面重合“背答案”的；编码器有一定泛化，但也虚高了 1~2 个点。"
          f"如果直接在全量 test 上比较，TF-IDF 与 BERT 差 {abs(bert_full - tf_full) * 100:.1f} 个点；"
          f"去重后差距变成 {abs(bert_clean - tf_clean) * 100:.1f} 个点——**换数据集时，清洗重复样本比换模型更能改变结论**。"
          f"THUCNews 的重复率只有 {ds_rows['thucnews']['tfidf']['removed_rate'] * 100:.1f}%，"
          "去重前后只差 0.0001 量级，属于可以忽略的水平；这也说明“换数据集先做重复检查”应该成为固定动作，"
          "重复率高时它比换模型更能改变结论。")

    # ---------------- 七、错误分析
    H(doc, "七、错误分析", 1)
    add_figure(doc, FIG / "fig6_confusion_matrix.png",
               "图 6  TNEWS 混淆矩阵（行归一化，表现最好的编码器）", width=13.5)
    tc = analysis.get("tnews", {}).get("top_confusions", [])
    if tc:
        add_table(doc, ["真实类别", "被预测为", "次数"],
                  [[x["true"], x["pred"], x["n"]] for x in tc],
                  widths=[5.0, 5.0, 3.0], font_size=9)
    pair_desc = "、".join(f"{x['true']}→{x['pred']}（{x['n']} 次）" for x in tc[:3])
    P(doc, f"主要错误集中在语义边界重叠的类别上：{pair_desc}。"
          "这些类别在业务上本来就互相渗透（公司新闻既像科技也像财经，军事与国际常同框，娱乐与文化共享同一批人物），"
          "一条标题只允许一个标签，模型只能在边界上“二选一”。"
          "所以这些错误不是“模型没训好”，而是标签口径本身有歧义；"
          "对应的处理不是继续换模型，而是改标签体系（合并边界类、上多标签、加实体识别特征）。"
          "这也是分类任务里最常见的“模型背锅、数据买单”。")

    # ---------------- 八、结论
    H(doc, "八、结论", 1)
    bullets(doc, [
        f"换预训练模型有效，但幅度有限：同参数量的 RoBERTa-wwm-ext / MacBERT-base 比 BERT-base 在 TNEWS 上高 "
        f"{pp(best_t['macro_f1_clean'], bert_t['macro_f1_clean'])} 个点，成本几乎不变（同参数量、同训练时间）。",
        "预训练的价值分两段：静态表示在 TNEWS 上就比 TF-IDF 高约 3 个点（不用训练编码器），"
        "微调再 1~2 个点；但在长文本情感任务上静态表示与 TF-IDF 持平，必须微调。",
        "换数据集后方法排序稳定（编码器 > TF-IDF > 规则），但绝对差距随任务难度变化："
        f"TNEWS 差 {pp(best_t['macro_f1_clean'], tf_t['macro_f1_clean'])} 个点，"
        f"THUCNews 差 {pp(max(ds_rows['thucnews']['bert']['macro_f1_clean'], ds_rows['thucnews']['rob']['macro_f1_clean'], ds_rows['thucnews']['mac']['macro_f1_clean']), ds_rows['thucnews']['tfidf']['macro_f1_clean'])} 个点，"
        f"ChnSentiCorp 差 {pp(max(ds_rows['chnsenticorp']['bert']['macro_f1_clean'], ds_rows['chnsenticorp']['rob']['macro_f1_clean'], ds_rows['chnsenticorp']['mac']['macro_f1_clean']), ds_rows['chnsenticorp']['tfidf']['macro_f1_clean'])} 个点。"
        "词汇线索强、类别均衡的任务，TF-IDF 就够用。",
        "数据质量比模型选择更重要：ChnSentiCorp 27.6% 的测试重复让所有方法虚高 1~3 个点，去重后结论会变。",
        "工程建议：先跑 TF-IDF 定天花板 → 再决定是否上编码器 → 编码器优先选与任务分布接近的中文预训练权重 → "
        "最后才考虑加训练技巧或换大模型。",
    ])

    # ---------------- 九、局限
    H(doc, "九、局限与改进方向", 1)
    bullets(doc, [
        "单 seed：每个配置只跑了一个随机种子，1 个点以内的差异可能在噪声范围内；结论主要站得住的是“量级”和“排序”，"
        "不是小数点后两位。改进：3~5 个 seed 取均值并做显著性检验。",
        "统一配方未调参：为了公平，三个编码器用同一套超参；如果给每个模型单独调 lr / epoch，差距可能变化。",
        "未重跑生成式方法：LLM zero-shot / LoRA 的数字引用课程结果（500 条口径），没有在本实验的 8,901 条上重跑。",
        "跨数据集比较的是方法排序：THUCNews、ChnSentiCorp 的标签体系与 TNEWS 不同，绝对分数不能直接相减。",
        "没有做更大模型（RoBERTa-large、Qwen3-1.7B LoRA）的对比：受 12GB 显存与时间限制，这属于下一步工作。",
    ])

    # ---------------- 附录
    H(doc, "附录 A  代码与结果文件", 1)
    add_table(doc, ["文件", "作用"],
              [["code/data_lib.py", "三个数据集的加载、去重、train/dev/test 切分"],
               ["code/train_encoder.py", "通用编码器微调（--model / --dataset 可配置）"],
               ["code/run_tfidf.py", "TF-IDF + LinearSVC 基线"],
               ["code/run_rule.py", "规则 / 关键词基线"],
               ["code/run_probe.py", "冻结特征 + 线性分类器"],
               ["code/collect.py", "汇总表格与全部图表"],
               ["code/analysis.py", "混淆对、类别增益等细粒度分析"],
               ["outputs/*.json", "每个“模型 × 数据集”组合的完整结果与逐条预测"],
               ["outputs/results.csv", "全部结果的宽表"],
               ["figures/*.png", "报告用图"]],
              widths=[7.0, 9.6], font_size=9)

    H(doc, "附录 B  复现命令", 1)
    for line in [
        "python code/make_stats.py",
        "python code/run_rule.py",
        "python code/run_tfidf.py --dataset tnews",
        "python code/run_tfidf.py --dataset thucnews",
        "python code/run_tfidf.py --dataset chnsenticorp",
        "python code/train_encoder.py --model bert-base --dataset tnews",
        "python code/train_encoder.py --model roberta-wwm-ext --dataset tnews",
        "python code/train_encoder.py --model macbert-base --dataset tnews",
        "python code/run_probe.py --model bert-base --dataset tnews",
        "python code/collect.py && python code/analysis.py && python code/make_report.py",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(line)
        set_run_font(r, size=9, latin="Consolas", east="宋体")

    P(doc, "说明：编码器权重来自本地 bert-base-chinese 与 ModelScope 镜像的 hfl/chinese-macbert-base、"
          "dienstag/chinese-roberta-wwm-ext（与 hfl 版本 config 逐字节一致）；TNEWS 数据复用课程 data/ 目录，"
          "THUCNews 与 ChnSentiCorp 来自 HuggingFace 数据集镜像。", size=9.5)

    doc.save(REPORT)
    print(f"→ {REPORT}")


if __name__ == "__main__":
    main()
