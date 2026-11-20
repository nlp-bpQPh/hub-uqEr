"""
标签体系与口径映射（全项目共用）

本轮改造说明：
  原数据是 TNEWS 商品分类（10 类：果汁/牙膏/白酒…），商品标题信息量小、
  词面特征太强，TF-IDF 就能到 0.9+，看不出模型之间的差异。
  现改为「中文多情绪对话」8 类分类：
    平静 / 开心 / 关心 / 生气 / 惊讶 / 伤心 / 厌恶 / 疑问
  情绪识别更依赖语义和上下文，不容易被单个关键词命中，指标区分度更好。

  对应数据：data/train.jsonl、data/test.jsonl（3321 / 830 条）
  字段：{"sentence": "文本", "label": 0~7, "category": "情绪名"}

设计要点：
  1. LABELS 的顺序就是 label 的整数编号，**必须和数据生成时一致**，
     改动会导致标签错位、所有指标失效。
  2. 下游脚本都用 len(LABELS) 决定输出层维度（num_labels），
     所以换类别体系只需要改这一张表，模型结构自动跟着变。
  3. 标签口径属于业务定义，写在代码里而不是模型里：换领域只改这张表。
  4. 规则法 / TF-IDF / BERT / LLM 四路方法的输出标签必须完全一致，
     否则同一个准确率在不同脚本里含义都不同，无法横向比较。
"""

# 8 类中文情绪（顺序即 label 编号，务必与 build_emotion_jsonl.py 的 EMOTIONS 一致）
LABELS = [
    "平静",   # 0
    "开心",   # 1
    "关心",   # 2
    "生气",   # 3
    "惊讶",   # 4
    "伤心",   # 5
    "厌恶",   # 6
    "疑问",   # 7
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# 情绪极性分组：8 个细类收拢成 3 个业务看得懂的桶
GROUP_MAP = {
    "开心": "正向情绪", "关心": "正向情绪",
    "平静": "中性情绪", "惊讶": "中性情绪", "疑问": "中性情绪",
    "生气": "负向情绪", "伤心": "负向情绪", "厌恶": "负向情绪",
}
GROUPS = ["正向情绪", "中性情绪", "负向情绪"]

# 情绪场景下每一类都有分析价值，不存在「噪音」类别，全部进视野
RELEVANT_GROUPS = set(GROUPS)


def to_group(label_id: int) -> str:
    """8 类情绪 id → 3 组极性"""
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    """8 类情绪 id → 是否与业务相关（情绪数据全部相关，保留接口以兼容下游）"""
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """
    分流结果落到哪个队列——这是「分类」变成「动作」的地方。

    设计要点：
      - 不相关 → drop（噪音，不进视野）
      - 相关但置信度低 → review（转 LLM 兜底 / 人工），宁可多一步也不要把不确定的推给人
      - 相关且高置信 → research（进研究队列）
    阈值 0.8 是默认值，实际要用真实 query 校准（见 service/classify_news.py 的说明）。
    """
    if not is_relevant(label_id):
        return "drop"
    if conf < conf_threshold:
        return "review"
    return "research"
