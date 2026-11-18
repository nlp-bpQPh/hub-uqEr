def display_menu():
    print("1.查看商品")
    print("2.加入购物车")
    print("3.查看购物车")
    print("4.计算总价")
    print("5.清空购物车")
    print("0.退出")

def show_products(products):
    for p in products:
        print(p["name"], p["price"])

def add_to_cart(cart, products, product_name, quantity):
    for p in products:
        if p["name"] == product_name:
            cart.append({"name":product_name,"price":p["price"],"num":quantity})
            return
    print("没有这个商品")

def show_cart(cart):
    for item in cart:
        print(item["name"], item["num"], item["price"])

def calculate_total(cart):
    total = 0
    for item in cart:
        total += item["price"] * item["num"]
    return total

def clear_cart(cart):
    cart.clear()

def main():
    products = [
        {"name":"苹果","price":5},
        {"name":"香蕉","price":3},
        {"name":"橙子","price":4}
    ]
    cart = []
    while True:
        display_menu()
        choice = input("请输入选择：")
        if choice == "1":
            show_products(products)
        elif choice == "2":
            name = input("输入商品名：")
            q = int(input("输入数量："))
            add_to_cart(cart, products, name, q)
        elif choice == "3":
            show_cart(cart)
        elif choice == "4":
            print("总价：", calculate_total(cart))
        elif choice == "5":
            clear_cart(cart)
            print("购物车已清空")
        elif choice == "0":
            break
        else:
            print("输入错误")

main()