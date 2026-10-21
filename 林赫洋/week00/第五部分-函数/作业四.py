def greet(name, greeting="你好", punctuation="！"):
    print(f"{greeting}，{name}{punctuation}")

def calculate_discount_price(price, discount=0.1, tax=0.0):
    return price * (1 - discount) * (1 + tax)

def create_student(name, age, grade="未定", city="未知"):
    return {
        "name": name,
        "age": age,
        "grade": grade,
        "city": city
    }

def print_info(title, items, separator=", ", end="\n"):
    info = ""
    for i, item in enumerate(items):
        if i == 0:
            info += item
        else:
            info += separator + item
    print(f"{title}: {info}{end}")

greet("owihof")
greet("hogweh", "你不好", "。。。")

print(f"折扣后的价格: {calculate_discount_price(100)}")
print(f"折扣后的价格（含税）: {calculate_discount_price(100, discount=0.2, tax=0.05)}")

student1 = create_student("张三", 20)
student2 = create_student("李四", 21, grade="大一", city="北京")

print_info("学生信息",[f"姓名: {student1['name']}", f"年龄: {student1['age']}", f"年级: {student1['grade']}", f"城市: {student1['city']}"])
print_info("学生信息",[f"姓名: {student2['name']}", f"年龄: {student2['age']}", f"年级: {student2['grade']}", f"城市: {student2['city']}"], separator=" | ")