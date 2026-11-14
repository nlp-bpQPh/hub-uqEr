name = input("请输入学生姓名：")
math = float(input("请输入数学成绩："))
chinese = float(input("请输入语文成绩："))
english = float(input("请输入英语成绩："))

total = math + chinese + english
average = total / 3

if average >= 90:
    grade = "优秀"
elif average >= 80:
    grade = "良好"
elif average >= 70:
    grade = "中等"
elif average >= 60:
    grade = "及格"
else:
    grade = "不及格"

print("姓名:", name)
print("数学:", math)
print("语文:", chinese)
print("英语:", english)
print("总分:", total)
print(f"平均分: {average:.2f}")
print("等级:", grade)

if math >= 60:
    print("数学及格")
else:
    print("数学不及格")

if chinese >= 60:
    print("语文及格")
else:
    print("语文不及格")

if english >= 60:
    print("英语及格")
else:
    print("英语不及格")
