# -*- coding: utf-8 -*-
# 作业二：用户输入验证 - 安全的用户输入处理


def get_positive_int(prompt="请输入一个正整数："):
    """获取正整数，输入不合法就重输，Ctrl+C 返回 None"""
    while True:
        try:
            n = int(input(prompt))
            if n <= 0:
                raise ValueError("必须是大于 0 的整数")
        except ValueError as e:
            print(f"  输入无效（{e}），请重新输入")
        except KeyboardInterrupt:
            print("\n  已取消输入")
            return None
        except EOFError:
            print("\n  输入结束")
            return None
        else:
            return n


def get_age(prompt="请输入年龄（0-150）："):
    """获取 0-150 的年龄"""
    while True:
        try:
            age = int(input(prompt))
            if age < 0:
                raise ValueError("年龄不能为负数")
            if age > 150:
                raise ValueError("年龄不能超过 150")
        except ValueError as e:
            print(f"  输入无效（{e}），请输入 0-150 的整数")
        except KeyboardInterrupt:
            print("\n  已取消输入")
            return None
        except EOFError:
            print("\n  输入结束")
            return None
        else:
            return age


def get_score(prompt="请输入分数（0-100，支持小数）："):
    """获取 0-100 的分数，支持小数"""
    while True:
        try:
            score = float(input(prompt))
            if score < 0 or score > 100:
                raise ValueError("分数必须在 0-100 之间")
        except ValueError as e:
            print(f"  输入无效（{e}），请重新输入")
        except KeyboardInterrupt:
            print("\n  已取消输入")
            return None
        except EOFError:
            print("\n  输入结束")
            return None
        else:
            return score


def main():
    print("===== 作业二：用户输入验证 =====")

    try:
        n = get_positive_int("  正整数：")
        age = get_age("  年龄：")
        score = get_score("  分数：")

        print("\n输入信息汇总：")
        print(f"  正整数：{n}")
        print(f"  年龄：{age}")
        print(f"  分数：{score}")
    except KeyboardInterrupt:
        print("\n程序已被用户中断")


if __name__ == "__main__":
    main()
