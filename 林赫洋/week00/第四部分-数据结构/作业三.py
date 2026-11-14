student = \
{
    "姓名": "张三",
    "年龄": 20,
    "学号": "1234567890",
    "数学成绩": 67,
    "语文成绩": 91,
    "英语成绩": 78
}

student["总分"] = student["数学成绩"] + student["语文成绩"] + student["英语成绩"]
student["平均分"] = student["总分"] / 3
if student["平均分"] >= 90:
    student["等级"] = "优秀"
elif student["平均分"] >= 80:
    student["等级"] = "良好"
elif student["平均分"] >= 70:
    student["等级"] = "中等"
elif student["平均分"] >= 60:
    student["等级"] = "及格"
else:
    student["等级"] = "不及格"

for key, value in student.items():
    print(f"{key}: {value}")

subject = input("请输入要修改的科目（数学成绩、语文成绩、英语成绩）：")
if subject in student:
    new_score = int(input(f"请输入新的{subject}："))
    student[subject] = new_score
    student["总分"] = student["数学成绩"] + student["语文成绩"] + student["英语成绩"]
    student["平均分"] = student["总分"] / 3
    if student["平均分"] >= 90:
        student["等级"] = "优秀"
    elif student["平均分"] >= 80:
        student["等级"] = "良好"
    elif student["平均分"] >= 70:
        student["等级"] = "中等"
    elif student["平均分"] >= 60:
        student["等级"] = "及格"
    else:
        student["等级"] = "不及格"

print("\n更新后的学生信息:")
for key, value in student.items():
    print(f"{key}: {value}")