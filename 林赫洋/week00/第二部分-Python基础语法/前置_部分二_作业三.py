student_id = "2404060703"
math_score = "100"
chinese_score = "90"
english_score = "80"

math_score = float(math_score) #字符串转换为浮点数
chinese_score = float(chinese_score)
english_score = float(english_score)

total_score = math_score + chinese_score + english_score
average_score = total_score / 3

total_score_str = str(round(total_score, 2)) #浮点数转换为字符串，两位小数
average_score_str = str(round(average_score, 2)) #round()四舍六入五成双

print(f"学号：{student_id}\n" +
      f"数学成绩：{math_score}\n" +
      f"语文成绩：{chinese_score}\n" +
      f"英语成绩：{english_score}\n" +
      f"总分：{total_score_str}\n" +
      f"平均分：{average_score_str}")