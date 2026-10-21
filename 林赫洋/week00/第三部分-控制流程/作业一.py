num = int(input("请输入一个整数："))

if num > 0:
    print(f"{num} 是正数")
elif num < 0:
    print(f"{num} 是负数")
else:
    print("0")

if num == 0:
    print("0")
elif num % 2 == 0:
    print(f"{num} 是偶数")
else:
    print(f"{num} 是奇数")

if num >= 0 and num <= 100:
    print(f"{num} 在 0 到 100 之间")
else:
    print(f"{num} 不在 0 到 100 之间")

