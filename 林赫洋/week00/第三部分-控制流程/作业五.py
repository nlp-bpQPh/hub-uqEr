choice = "0"

while choice != "5":
    print("1. 加法")
    print("2. 减法")
    print("3. 乘法")
    print("4. 除法")
    print("5. 退出")
    choice = input("请选择：")

    if choice == "1":
        a = float(input("请输入第一个数字："))
        b = float(input("请输入第二个数字："))
        print(f"{a} + {b} = {a + b}")
    elif choice == "2":
        a = float(input("请输入第一个数字："))
        b = float(input("请输入第二个数字："))
        print(f"{a} - {b} = {a - b}")
    elif choice == "3":
        a = float(input("请输入第一个数字："))
        b = float(input("请输入第二个数字："))
        print(f"{a} * {b} = {a * b}")
    elif choice == "4":
        a = float(input("请输入被除数："))
        b = float(input("请输入除数："))
        if b == 0:
            print("除数不能为0")
        else:
            print(f"{a} / {b} = {a / b}")
    elif choice == "5":
        print("退出")
    else:
        print("输入无效")
