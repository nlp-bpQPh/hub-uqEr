students = []
for i in range(5):
    name = input(f"请输入第{i+1}个学生的姓名: ")
    students.append(name)

for student in students:
    print(student,end=" ")

print(f"\n学生总数: {len(students)}")

search_name = input("请输入要查找的学生姓名: ")
if search_name in students:
    print(f"{search_name} 在列表中的索引是: {students.index(search_name)}")
else:
    print(f"{search_name} 不在列表中")