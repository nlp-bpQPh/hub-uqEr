import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
'''

关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models

'''

# bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
# 预训练模型路径：默认用课件目录下的 pretrain_models，可用环境变量 BERT_DIR 覆盖
import os
BERT_DIR = os.environ.get("BERT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "pretrain_models", "bert-base-chinese"))
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)

string = "咱呀么老百姓今儿个真高兴"
#分字
tokens = tokenizer.tokenize(string)
print("分字：", tokens)
#编码，前后自动添加了[cls] string [sep]
encoding = tokenizer.encode(string)
print("编码：", encoding)
#文本对编码, 形式[cls] string1 [sep] string2 [sep]
string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)



