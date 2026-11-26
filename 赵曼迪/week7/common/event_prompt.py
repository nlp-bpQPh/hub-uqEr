def build_system_prompt(schema: dict) -> str:
    """
    schema: dict, 格式 {事件类型名: [角色列表,...]}
    由 load_schema() 返回，过滤掉 __enums__ 等下划线开头key
    """
    lines = [
        "你是专业的公告事件抽取助手，请严格按照规则从给定句子中抽取事件。",
        "规则：",
        "1. 只输出JSON结果，不要输出任何解释、说明、多余中文。",
        "2. trigger：填写句子中触发该事件的原始词语，不能为空。",
        "3. arguments的text必须直接摘抄原文文本，禁止编造、改写文本。",
        "4. 只能使用下面列出的事件类型与角色，禁止自定义事件类型、自定义角色。",
        "5. 一个句子可能包含0个、1个或多个事件；没有识别到事件输出 {\"events\":[]}。",
        "",
        "输出JSON格式示例：",
        '{"events": [{"type":"事件类型","trigger":{"text":"触发词"},"arguments":[{"role":"角色","text":"论元文本"}]}]}',
        "",
        "支持的事件类型以及对应的角色："
    ]

    for evt_type, role_list in schema.items():
        roles_str = "、".join(role_list)
        lines.append(f"- 事件类型：{evt_type}，可填角色：{roles_str}")

    lines.append("\n输出：")
    return "\n".join(lines)