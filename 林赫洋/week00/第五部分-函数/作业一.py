def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b
def calculate_bmi(weight, height):
    return weight / (height ** 2)

a = float(input("请输入第一个数字: "))
b = float(input("请输入第二个数字: "))
print(f"{a} + {b} = {add(a, b)}")
print(f"{a} - {b} = {subtract(a, b)}")
print(f"{a} * {b} = {multiply(a, b)}")
print(f"{a} / {b} = {divide(a, b)}")
weight = float(input("请输入体重(kg): "))
height = float(input("请输入身高(m): "))
print(f"BMI = {calculate_bmi(weight, height)}")