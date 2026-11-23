class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, name, price, count=1):
        good = {"name": name, "price": price, "count": count}
        self.items.append(good)

    def remove_item(self, name):
        for item in self.items:
            if item["name"] == name:
                self.items.remove(item)
                print(f"已移除商品：{name}")
                return
        print(f"购物车中没有商品：{name}")

    def calculate_total(self):
        total = 0
        for item in self.items:
            total += item["price"] * item["count"]
        return total

    def display_cart(self):
        if len(self.items) == 0:
            print("购物车是空的")
            return
        for item in self.items:
            print(f"商品：{item['name']}，单价：{item['price']}，数量：{item['count']}")
        print(f"当前总价：{self.calculate_total()}")

    def clear_cart(self):
        self.items.clear()
        print("购物车已清空！")


cart = ShoppingCart()

cart.add_item("笔记本", 10, 2)
cart.add_item("钢笔", 15, 1)
cart.add_item("橡皮", 2, 3)

cart.display_cart()

cart.remove_item("钢笔")

cart.display_cart()
