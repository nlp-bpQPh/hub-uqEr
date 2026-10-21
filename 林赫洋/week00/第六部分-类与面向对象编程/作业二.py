class BankAccount:
    def __init__(self, account_id, name, balance=0):
        self.account_id = account_id
        self.name = name
        self.balance = balance

    def deposit(self, money):
        self.balance += money
        print(f"成功存入{money}元")

    def withdraw(self, money):
        if self.balance >= money:
            self.balance -= money
            print(f"成功取出{money}元")
        else:
            print("余额不足，取款失败！")

    def get_balance(self):
        return self.balance

    def display_info(self):
        print(f"账号：{self.account_id}，户名：{self.name}，当前余额：{self.balance}元")


account = BankAccount("6222001", "小明")

account.deposit(1000)
account.withdraw(300)
account.withdraw(800)

account.display_info()
