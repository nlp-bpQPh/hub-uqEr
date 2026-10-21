things = [
    {"商品名称": "苹果", "价格": 3.5, "库存数量": 100},
    {"商品名称": "香蕉", "价格": 2.0, "库存数量": 150},
    {"商品名称": "橙子", "价格": 4.0, "库存数量": 80},
    {"商品名称": "西瓜", "价格": 6.0, "库存数量": 50},
    {"商品名称": "葡萄", "价格": 5.0, "库存数量": 120}
]
car = []
key = input("请输入操作编号(1-6)：")
while key != "6":
    if key == "1":
        for t in things:
            print(f"商品名称：{t['商品名称']}，价格：{t['价格']}元，库存数量：{t['库存数量']}")
    elif key == "2":
        name = input("请输入要添加的商品名称：")
        for t in things:
            if t["商品名称"] == name:
                num = int(input("请输入购买数量："))
                if num <= t["库存数量"]:
                    car.append({"商品名称": name, "购买数量": num, "单价": t["价格"]})
                    t["库存数量"] -= num
                    print(f"已将 {num} 个 {name} 添加到购物车。")
                else:
                    print("库存不足")
                break
        else:
            print("商品不存在")
    elif key == "3":
        for c in car:
            print(f"商品名称：{c['商品名称']}，购买数量：{c['购买数量']}，单价：{c['单价']}元")
    elif key == "4":
        total = sum(c["购买数量"] * c["单价"] for c in car)
        print(f"购物车总价：{total:.2f}元")
    elif key == "5":
        car.clear()
        print("购物车已清空。")
    else:
        print("无效操作，请重新输入。")
    
    key = input("请输入操作编号(1-6)：")