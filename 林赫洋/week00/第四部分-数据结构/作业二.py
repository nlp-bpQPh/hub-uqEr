# 任务描述：
# 编写一个成绩管理系统，实现以下功能：
# 1. 创建一个列表存储5个学生的成绩（可以预设或用户输入）
# 2. 计算并输出：总分、平均分、最高分、最低分
# 3. 统计及格人数（成绩>=60）和不及格人数
# 4. 找出所有大于等于90分的成绩，并输出
# 5. 使用循环遍历列表，为每个成绩评定等级：
#    - 90分及以上：优秀
#    - 80-89分：良好
#    - 70-79分：中等
#    - 60-69分：及格
#    - 60分以下：不及格
# 6. 输出格式化的成绩单
#
# 要求：
# - 使用列表的索引访问元素
# - 使用for循环遍历列表
# - 使用条件语句进行判断
# - 使用列表的append()方法（如果需要）
# - 输出格式要清晰美观

scores = [85, 92, 78, 99, 65]
total = sum(scores)
average = total / len(scores)
max_score = max(scores)
min_score = min(scores)

print(f"总分: {total}", f"平均分: {average:.2f}", f"最高分: {max_score}", f"最低分: {min_score}")

pass_count = 0
fail_count = 0
for score in scores:
    if score >= 60:
        pass_count += 1
    else:
        fail_count += 1
print(f"及格人数: {pass_count}", f"不及格人数: {fail_count}")

for score in scores:
    if score >= 90:
        print(f"{score} 优秀")
    elif score >= 80:
        print(f"{score} 良好")
    elif score >= 70:
        print(f"{score} 中等")
    elif score >= 60:
        print(f"{score} 及格")
    else:
        print(f"{score} 不及格")

print("\n成绩单:")
for i, score in enumerate(scores, start=1):
    print(f"学生{i}: {score}分")
