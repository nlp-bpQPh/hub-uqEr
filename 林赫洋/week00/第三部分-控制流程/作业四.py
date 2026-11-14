for i in range(1, 21):
    print(i, end=" ")
print()

for i in range(2, 21, 2):
    print(i, end=" ")
print()

for i in range(1, 21, 2):
    print(i, end=" ")
print()

for i in range(20, 0, -1):
    print(i, end=" ")
print()

product = 1
for i in range(1, 11):
    product *= i
print("1到10的乘积：", product)

text = input("请输入一个字符串：")
target = input("请输入要查找的字符：")

count = 0
for ch in text:
    if ch == target:
        count += 1

print(f"字符 {target} 出现了 {count} 次")
