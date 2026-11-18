n = int(input("请输入一个正整数n："))

total = 0
even_sum = 0
odd_sum = 0
i = 1

while i <= n:
    total += i
    if i % 2 == 0:
        even_sum += i
    else:
        odd_sum += i
    i += 1

print(f"1到{n}的和：{total}")
print(f"1到{n}的偶数和：{even_sum}")
print(f"1到{n}的奇数和：{odd_sum}")
