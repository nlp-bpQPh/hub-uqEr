name_A = input("输入A的名字：")
price_A = float(input("输入A的价格："))
count_A = int(input("输入A的数量："))

name_B = input("输入B的名字：")
price_B = float(input("输入B的价格："))
count_B = int(input("输入B的数量："))

name_C = input("输入C的名字：")
price_C = float(input("输入C的价格："))
count_C = int(input("输入C的数量："))

sum_A = price_A * count_A
sum_B = price_B * count_B
sum_C = price_C * count_C

total_sum = sum_A + sum_B + sum_C

print(f"商品A：{name_A}，单价：{price_A}元，数量：{count_A}，总价：{sum_A}元\n" +
      f"商品B：{name_B}，单价：{price_B}元，数量：{count_B}，总价：{sum_B}元\n" +
      f"商品C：{name_C}，单价：{price_C}元，数量：{count_C}，总价：{sum_C}元\n" +
      f"总价：{total_sum}元")

#若打9折
discounted_total_sum = total_sum * 0.9
print(f"折扣后的总价：{discounted_total_sum}元")