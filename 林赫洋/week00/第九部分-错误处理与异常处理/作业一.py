# -*- coding: utf-8 -*-
# 作业一：基础异常处理 - 安全的数值计算


def safe_divide(a, b):
    """安全除法，出错返回 None"""
    try:
        result = a / b
    except ZeroDivisionError:
        print("错误：除数不能为 0")
        return None
    except TypeError:
        print("错误：除法运算的参数必须是数字")
        return None
    else:
        print(f"{a} / {b} = {result}")
        return result
    finally:
        print("  safe_divide 执行结束")


def safe_power(base, exp):
    """安全幂运算，出错返回 None"""
    try:
        if not isinstance(base, (int, float)) or not isinstance(exp, (int, float)):
            raise TypeError("底数和指数必须是数字")
        result = base ** exp
    except TypeError as e:
        print(f"错误：{e}")
        return None
    else:
        print(f"{base} ** {exp} = {result}")
        return result
    finally:
        print("  safe_power 执行结束")


def main():
    print("===== 作业一：安全的数值计算 =====")

    print("\n【测试 safe_divide】")
    print("正常：", safe_divide(10, 2))
    print("正常：", safe_divide(7, 3))
    print("除数为 0：", safe_divide(10, 0))
    print("类型错误：", safe_divide(10, "abc"))
    print("类型错误：", safe_divide(None, 2))

    print("\n【测试 safe_power】")
    print("正常：", safe_power(2, 10))
    print("正常：", safe_power(2, 0.5))
    print("类型错误：", safe_power("2", 3))
    print("类型错误：", safe_power(2, [1, 2]))

    print("\n【批量计算：出错也不中断】")
    for a, b in [(100, 5), (100, 0), (8, 2), ("a", 1), (15, 4)]:
        r = safe_divide(a, b)
        if r is None:
            print("  本次失败，跳过")
        else:
            print(f"  结果：{r}")


if __name__ == "__main__":
    main()
