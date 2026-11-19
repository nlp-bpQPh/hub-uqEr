"""校验生成的 DOCX：结构、表格、图片、关键数字。"""

import re
import sys

from docx import Document

PATH = r"E:\大模型学习\作业\第六周作业-分类方法对比\报告_第六周_分类方法对比实验.docx"

doc = Document(PATH)
paras = [p for p in doc.paragraphs if p.text.strip()]
print(f"段落(非空): {len(paras)}  表格: {len(doc.tables)}  内嵌图片: {len(doc.inline_shapes)}")

heads = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
print("标题:")
for h in heads:
    print("  ", h)

text = "\n".join(p.text for p in doc.paragraphs)
tables_text = "\n".join(
    "\t".join(c.text for c in row.cells) for t in doc.tables for row in t.rows)
all_text = text + "\n" + tables_text

checks = {
    "摘要": "摘要" in text,
    "实验一": "实验一" in text,
    "实验二": "实验二" in text,
    "结论": "结论" in text,
    "0.5508 / 0.5494": ("0.5508" in all_text and "0.5494" in all_text),
    "ChnSenti 0.9423": "0.9423" in all_text,
    "27.6%": "27.6%" in all_text,
    "无占位符 {}": "{" not in all_text,
    "无 None": "None" not in all_text,
    "无 NaN": "nan" not in all_text.lower(),
}
print("\n检查:")
for k, v in checks.items():
    print(f"  [{'OK' if v else 'FAIL'}] {k}")

print("\n字数(正文):", len(re.sub(r"\s", "", text)))
