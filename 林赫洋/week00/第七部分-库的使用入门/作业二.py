import datetime

now = datetime.datetime.now()
print("获取当前日期和时间：", now)
print("格式化显示日期和时间：", now.strftime("%Y-%m-%d %H:%M:%S"))
d1 = datetime.datetime(2025, 1, 1)
d2 = datetime.datetime(2026, 1, 1)
print("两个日期之间的天数差：", (d2 - d1).days)
target_day = d1 + datetime.timedelta(days=10)
print("某个日期之后N天的日期：", target_day)
week_list = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
print("今天是星期几：", week_list[now.weekday()])