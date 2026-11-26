"""
标签体系与安全分流口径（全项目共用）

教学重点：
  1. 公开数据集是「替代数据」——malicious_phish 的 4 个 URL 安全类别，和业务关心的
     「放行/拦截」动作不是一回事；所以第一步不是训练模型，而是**定义标签口径**：
     4 类重映射成 3 组（safe / phishing / malicious）+ 「是否需要处理」派生。
  2. 「是否需要处理」由主题组派生（phishing + malicious = 需要拦），**零人工标注**——
     这解释了为什么「换个领域做分类」往往不需要重新标数据，改映射表就行。
  3. 标签口径属于业务定义，写在代码里而不是模型里：换领域只改这张表。

为什么单独成文件：
  规则法、TF-IDF、BERT、LLM 四路方法的输出标签必须完全一致，
  否则同一个 0.9 的准确率在不同脚本里含义都不同，无法横向比较。
"""

# malicious_phish.csv 的 4 类（来自 type 字段；顺序固定，idx 依赖它）
LABELS = ["benign", "phishing", "malware", "defacement"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# 3 组主题：safe（正常）/ phishing（凭证窃取）/ malicious（站点被黑或投毒）
GROUP_MAP = {
    "benign": "safe",
    "phishing": "phishing",
    "malware": "malicious",
    "defacement": "malicious",
}
GROUPS = ["safe", "phishing", "malicious"]

# 安全团队关心「要不要拦」：phishing 与 malicious 组都需要处理；safe 直接放行
RELEVANT_GROUPS = {"phishing", "malicious"}


def to_group(label_id: int) -> str:
    """4 类 id → 3 组主题"""
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    """4 类 id → 是否需要安全处置（由主题组派生，不需要额外标注）"""
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """
    分流结果落到哪个队列——这是「分类」变成「动作」的地方。

    设计要点：
      - 良性 → pass（放行，不进安全视野）
      - 疑似恶意但置信度低 → review（转人工复核），宁可多一步也不要放过不确定的
      - 恶意且高置信 → block（直接拦截）
    阈值 0.8 是默认值，实际要用真实流量校准。
    """
    if not is_relevant(label_id):
        return "pass"
    if conf < conf_threshold:
        return "review"
    return "block"
