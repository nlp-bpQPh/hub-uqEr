class Student:
    def __init__(self, name, age, student_id):
        self.name = name
        self.age = age
        self.student_id = student_id
        self.math = 0
        self.chinese = 0
        self.english = 0

    def display_info(self):
        avg = self.calculate_average()
        grade = self.get_grade()
        print(f"学号：{self.student_id}，姓名：{self.name}，年龄：{self.age}")
        print(f"语文：{self.chinese}，数学：{self.math}，英语：{self.english}")
        print(f"平均分：{avg:.2f}，等级：{grade}\n")

    def set_score(self, chinese, math, english):
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


s1 = Student("张三", 18, "2026001")
s2 = Student("李四", 17, "2026002")

s1.set_score(88, 92, 79)
s2.set_score(55, 66, 70)

s1.display_info()
s2.display_info()
