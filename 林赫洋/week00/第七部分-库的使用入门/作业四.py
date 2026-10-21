import random

try:
    with open("student.txt", "r", encoding="utf-8") as f:
        names = f.readlines()
    # 清除换行符
    names = [student.strip() for student in names if student.strip()]
except FileNotFoundError:
    print("参与者文件不存在！")
else:
    # 设置获奖人数
    win_count = 1
    # 随机抽取，不重复
    winners = random.sample(names, win_count)
    print("获奖名单：", winners)
    # 写入获奖名单
    with open("winner.txt", "w", encoding="utf-8") as f:
        for w in winners:
            f.write(w + "\n")