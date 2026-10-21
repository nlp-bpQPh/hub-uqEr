class Student:
    def __init__(self, name, age, student_id, chinese, math, english):
        self.name = name
        self.age = age
        self.student_id = student_id
        self.chinese = chinese
        self.math = math
        self.english = english

    def calculate_average(self):
        total = self.chinese + self.math + self.english
        return total / 3

    def get_grade(self):
        avg = self.calculate_average()
        if avg >= 90:
            return "A"
        elif avg >= 80:
            return "B"
        elif avg >= 70:
            return "C"
        elif avg >= 60:
            return "D"
        else:
            return "E"

    def display_info(self):
        avg = self.calculate_average()
        grade = self.get_grade()
        print(f"学号:{self.student_id} 姓名:{self.name} 年龄:{self.age}")
        print(f"语文:{self.chinese} 数学:{self.math} 英语:{self.english} 平均分:{avg:.2f} 等级:{grade}")


class StudentManager:
    def __init__(self):
        self.student_list = []

    def add_student(self, student_obj):
        self.student_list.append(student_obj)

    def find_student(self, target_id):
        for stu in self.student_list:
            if stu.student_id == target_id:
                return stu
        return None

    def display_all(self):
        for stu in self.student_list:
            stu.display_info()
            print("-" * 30)

    def get_class_average(self):
        if len(self.student_list) == 0:
            return 0
        sum_total = 0
        for stu in self.student_list:
            sum_total += stu.calculate_average()
        return sum_total / len(self.student_list)


manager = StudentManager()

s1 = Student("张三", 18, "202601", 85, 90, 78)
s2 = Student("李四", 17, "202602", 60, 72, 68)
s3 = Student("王五", 18, "202603", 92, 95, 88)

manager.add_student(s1)
manager.add_student(s2)
manager.add_student(s3)

manager.display_all()

find_result = manager.find_student("202602")
if find_result:
    find_result.display_info()
else:
    print("未找到该学生")

class_avg = manager.get_class_average()
print(f"\n班级整体平均分：{class_avg:.2f}")
