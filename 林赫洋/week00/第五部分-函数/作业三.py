def create_student(name, age, scores):
    return {
        "name": name,
        "age": age,
        "scores": scores,
        "average": 0,
        "grade": ""
    }

def calculate_student_average(student):
    student["average"] = sum(student["scores"]) / len(student["scores"])

def get_student_grade(student):
    if student["average"] >= 90:
        student["grade"] = "优秀"
    elif student["average"] >= 80:
        student["grade"] = "良好"
    elif student["average"] >= 70:
        student["grade"] = "中等"
    elif student["average"] >= 60:
        student["grade"] = "及格"
    else:
        student["grade"] = "不及格"

def display_student_info(student):
    print(f"姓名: {student['name']}, 年龄: {student['age']}, 平均分: {student['average']:.2f}, 等级: {student['grade']}")

def add_student(students_list, name, age, scores):
    student = create_student(name, age, scores)
    calculate_student_average(student)
    get_student_grade(student)
    students_list.append(student)

def find_student(students_list, name):
    for student in students_list:
        if student["name"] == name:
            return student
    return None

def get_class_average(students_list):
    total_average = sum(student["average"] for student in students_list)
    return total_average / len(students_list) if students_list else 0

students = []
add_student(students, "AAA", 20, [85, 92, 78])
add_student(students, "BBB", 21, [75, 88, 90])
add_student(students, "CCC", 22, [65, 70, 75])
for student in students:
    display_student_info(student)

student = input("请输入要查找的学生姓名: ")
student = find_student(students, student)
if student:
    display_student_info(student)
else:
    print(f"未找到学生: {student}")

class_average = get_class_average(students)
print(f"班级平均分: {class_average:.2f}")