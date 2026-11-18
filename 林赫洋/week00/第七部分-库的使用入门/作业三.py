students = [
    "张三,20,85,90,88",
    "李四,19,92,88,90",
    "王五,21,78,82,86"
]

#写入
with open("student.txt", "w", encoding="utf-8") as f:
    for s in students:
        f.write(s + "\n")

#读取
with open("student.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

#解析
print("解析学生信息：")
for line in lines:
    line = line.strip()
    info = line.split(",")
    print(info)

#追加
new_stu = "赵六,20,88,85,91"
with open("student.txt", "a", encoding="utf-8") as f:
    f.write(new_stu + "\n")