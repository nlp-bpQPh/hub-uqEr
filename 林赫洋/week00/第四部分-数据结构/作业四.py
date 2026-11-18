students = [
    {"姓名": "张三", "年龄": 20, "数学": 85, "语文": 90, "英语": 88},
    {"姓名": "李四", "年龄": 19, "数学": 92, "语文": 88, "英语": 90},
    {"姓名": "王五", "年龄": 21, "数学": 78, "语文": 85, "英语": 82}
]

# 计算每个人平均分并存入字典
for s in students:
    s["平均分"] = (s["数学"] + s["语文"] + s["英语"]) / 3

# 打印全部学生信息
for s in students:
    print(f"姓名：{s['姓名']}，年龄：{s['年龄']}岁")
    print(f"数学：{s['数学']}，语文：{s['语文']}，英语：{s['英语']}，平均分：{s['平均分']:.2f}\n")

# 找最高分学生
max_avg = students[0]["平均分"]
best = students[0]
for s in students:
    if s["平均分"] > max_avg:
        max_avg = s["平均分"]
        best = s

# 全班总平均分
total = 0
for s in students:
    total += s["平均分"]
class_avg = total / 3
print(f"全班平均分：{class_avg:.2f}")
print(f"平均分最高：{best['姓名']}，{best['平均分']:.2f}\n")

# 等级统计
ex = good = mid = pa = fail = 0
for s in students:
    a = s["平均分"]
    if a >=90:
        ex +=1
    elif a >=80:
        good +=1
    elif a >=70:
        mid +=1
    elif a >=60:
        pa +=1
    else:
        fail +=1
print(f"优秀：{ex}，良好：{good}，中等：{mid}，及格：{pa}，不及格：{fail}\n")

# 查找+修改
name = input("输入查找姓名：")
for s in students:
    if s["姓名"] == name:
        print("\n找到：", s)
        sub = input("修改科目(数学/语文/英语)：")
        if sub in ["数学","语文","英语"]:
            newsc = float(input("输入新分数："))
            s[sub] = newsc
            s["平均分"] = (s["数学"] + s["语文"] + s["英语"])/3
            print("更新后：", s)
        else:
            print("科目错误")
        break
else:
    print("未找到")