"""两阶段流水线示例文本。"""

EXAMPLES = [
    {
        "id": "pipeline_01",
        "text": "万讯自控公告称，股东傅宇晨办理了股份质押，质押数量为2290万股。",
        "event": {
            "type": "质押",
            "trigger": {"text": "质押", "start": 18, "end": 20},
            "args": [
                {"role": "质押物所属公司", "text": "万讯自控", "start": 0, "end": 4},
                {"role": "质押方", "text": "傅宇晨", "start": 10, "end": 13},
                {"role": "质押物", "text": "股份", "start": 16, "end": 18},
                {
                    "role": "质押股票/股份数量",
                    "text": "2290万",
                    "start": 26,
                    "end": 31,
                },
            ],
        },
    },
    {
        "id": "pipeline_02",
        "text": "众安集团于2019年10月15日回购380.5万股，交易金额为94.56万港元。",
        "event": {
            "type": "股份回购",
            "trigger": {"text": "回购", "start": 16, "end": 18},
            "args": [
                {"role": "回购方", "text": "众安集团", "start": 0, "end": 4},
                {
                    "role": "回购完成时间",
                    "text": "2019年10月15日",
                    "start": 5,
                    "end": 16,
                },
                {
                    "role": "回购股份数量",
                    "text": "380.5万",
                    "start": 18,
                    "end": 24,
                },
                {
                    "role": "交易金额",
                    "text": "94.56万港元",
                    "start": 31,
                    "end": 39,
                },
            ],
        },
    },
    {
        "id": "pipeline_03",
        "text": "国家电网公布中标结果，公司中标智能电表采购项目，金额约为1.85亿元。",
        "event": {
            "type": "中标",
            "trigger": {"text": "中标", "start": 6, "end": 8},
            "args": [
                {"role": "招标方", "text": "国家电网", "start": 0, "end": 4},
                {
                    "role": "中标标的",
                    "text": "智能电表采购项目",
                    "start": 15,
                    "end": 23,
                },
                {
                    "role": "中标金额",
                    "text": "1.85亿元",
                    "start": 28,
                    "end": 34,
                },
            ],
        },
    },
    {
        "id": "pipeline_04",
        "text": "永辉超市拟聘任金斌为副总裁，该事项于7月31日披露。",
        "event": {
            "type": "高管变动",
            "trigger": {"text": "聘任", "start": 5, "end": 7},
            "args": [
                {"role": "任职公司", "text": "永辉超市", "start": 0, "end": 4},
                {"role": "高管姓名", "text": "金斌", "start": 7, "end": 9},
                {
                    "role": "变动后职位",
                    "text": "副总裁",
                    "start": 10,
                    "end": 13,
                },
                {"role": "披露日期", "text": "7月31日", "start": 18, "end": 23},
            ],
        },
    },
    {
        "id": "pipeline_05",
        "text": "鼎胜新材上半年净亏损1254.97万元，上年同期为盈利。",
        "event": {
            "type": "亏损",
            "trigger": {"text": "亏损", "start": 8, "end": 10},
            "args": [
                {"role": "公司名称", "text": "鼎胜新材", "start": 0, "end": 4},
                {"role": "财报周期", "text": "上半年", "start": 4, "end": 7},
                {
                    "role": "净亏损",
                    "text": "1254.97万元",
                    "start": 10,
                    "end": 19,
                },
                {
                    "role": "亏损变化",
                    "text": "上年同期为盈利",
                    "start": 20,
                    "end": 27,
                },
            ],
        },
    },
]

