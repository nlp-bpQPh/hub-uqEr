# -*- coding: utf-8 -*-
# 作业四：综合应用 - 学生成绩管理系统（异常处理版）

import json
import os


class StudentScoreManager:
    """成绩管理器，数据格式：{"张三": [85, 92.5]}"""

    def __init__(self, data_file="scores.json"):
        self.data_file = data_file
        self.scores = {}
        self.load_scores()

    def load_scores(self):
        """从 JSON 文件加载成绩"""
        try:
            if not os.path.exists(self.data_file):
                print(f"提示：'{self.data_file}' 不存在，使用空数据启动")
                return False

            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise TypeError("数据格式错误，期望字典")
            for name, lst in data.items():
                if not isinstance(lst, list):
                    raise TypeError(f"'{name}' 的成绩不是列表")
                for s in lst:
                    if not isinstance(s, (int, float)):
                        raise TypeError(f"'{name}' 的成绩不是数字")

        except json.JSONDecodeError as e:
            print(f"错误：'{self.data_file}' 格式损坏，{e}")
            self.scores = {}
            return False
        except PermissionError:
            print(f"错误：没有权限读取 '{self.data_file}'")
            self.scores = {}
            return False
        except (TypeError, ValueError) as e:
            print(f"错误：数据内容不合法，{e}")
            self.scores = {}
            return False
        except OSError as e:
            print(f"错误：读取失败，{e}")
            self.scores = {}
            return False
        else:
            self.scores = data
            print(f"加载成功：{len(self.scores)} 名学生")
            return True

    def save_scores(self):
        """保存成绩到 JSON 文件"""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.scores, f, ensure_ascii=False, indent=4)
        except PermissionError:
            print(f"错误：没有权限写入 '{self.data_file}'")
            return False
        except FileNotFoundError:
            print(f"错误：路径 '{self.data_file}' 不存在")
            return False
        except TypeError as e:
            print(f"错误：数据无法序列化，{e}")
            return False
        except OSError as e:
            print(f"错误：保存失败，{e}")
            return False
        else:
            print(f"保存成功：{len(self.scores)} 名学生已写入 '{self.data_file}'")
            return True

    def add_score(self, name, score):
        """添加成绩，验证姓名和 0-100 的分数"""
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("姓名必须是非空字符串")
            name = name.strip()

            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("成绩必须是数字")
            if score < 0 or score > 100:
                raise ValueError(f"成绩 {score} 超出 0-100")
        except (ValueError, TypeError) as e:
            print(f"错误：添加失败（{name}），{e}")
            return False
        else:
            self.scores.setdefault(name, []).append(float(score))
            print(f"添加成功：{name} - {score}")
            return True

    def get_average(self, name):
        """计算平均分，失败返回 None"""
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("姓名必须是非空字符串")
            name = name.strip()
            if name not in self.scores:
                raise KeyError(f"学生 '{name}' 不存在")
            lst = self.scores[name]
            if not lst:
                raise ValueError(f"'{name}' 暂无成绩")
            avg = sum(lst) / len(lst)
        except (ValueError, KeyError) as e:
            print(f"错误：{e}")
            return None
        except ZeroDivisionError:
            print(f"错误：'{name}' 成绩列表为空")
            return None
        else:
            print(f"{name} 的平均分：{avg:.2f}")
            return avg

    def display_all(self):
        """显示所有学生成绩"""
        print("\n学生成绩总览")
        print("-" * 30)
        if not self.scores:
            print("暂无成绩记录")
            return
        for name, lst in self.scores.items():
            if lst:
                print(f"  {name}：{lst}，平均分：{sum(lst) / len(lst):.2f}")
            else:
                print(f"  {name}：暂无成绩")
        print("-" * 30)


def main():
    print("===== 作业四：学生成绩管理系统 =====")

    manager = StudentScoreManager("scores.json")

    print("\n【添加成绩：正常】")
    manager.add_score("张三", 85)
    manager.add_score("张三", 92.5)
    manager.add_score("李四", 78)
    manager.add_score("王五", 66)

    print("\n【添加成绩：异常】")
    manager.add_score("赵六", 120)
    manager.add_score("赵六", -5)
    manager.add_score("赵六", "优秀")
    manager.add_score("", 90)
    manager.add_score("孙七", None)

    print("\n【查询平均分】")
    manager.get_average("张三")
    manager.get_average("李四")
    manager.get_average("不存在的人")
    manager.get_average("")

    print("\n【显示所有成绩】")
    manager.display_all()

    print("\n【保存并重新加载】")
    manager.save_scores()
    manager2 = StudentScoreManager("scores.json")
    manager2.display_all()
    manager2.get_average("张三")


if __name__ == "__main__":
    main()
