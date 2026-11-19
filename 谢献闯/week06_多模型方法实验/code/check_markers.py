"""检查 DOCX 里是否残留 Markdown 标记或异常字符。"""

from docx import Document

PATH = r"E:\大模型学习\作业\第六周作业-分类方法对比\报告_第六周_分类方法对比实验.docx"
doc = Document(PATH)

texts = [p.text for p in doc.paragraphs]
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            texts.append(c.text)

bad = [t for t in texts if "**" in t or "\\*" in t or "{" in t or "}" in t]
print("含可疑标记的文本块:", len(bad))
for b in bad[:5]:
    print("  ", b[:120])

print("总文本块:", len(texts))
print("表格:", len(doc.tables), "图片:", len(doc.inline_shapes))
