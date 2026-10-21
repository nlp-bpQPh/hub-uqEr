def calculate_average(scores):
    return sum(scores) / len(scores)

def find_max_min(scores):
    return max(scores), min(scores)

def count_pass_fail(scores, pass_score=60):
    pass_count = 0
    fail_count = 0
    for score in scores:
        if score >= pass_score:
            pass_count += 1
        else:
            fail_count += 1
    return pass_count, fail_count

def get_grade(score):
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "中等"
    elif score >= 60:
        return "及格"
    else:
        return "不及格"

def process_scores(scores):
    return {
        "average": calculate_average(scores),
        "max": find_max_min(scores)[0],
        "min": find_max_min(scores)[1],
        "pass_count": count_pass_fail(scores)[0],
        "fail_count": count_pass_fail(scores)[1]
    }

scores = [85, 92, 78, 99, 65]
result = process_scores(scores)
print(f"总分: {result['average']}")
print(f"最高分: {result['max']}")
print(f"最低分: {result['min']}")
print(f"及格人数: {result['pass_count']}")
print(f"不及格人数: {result['fail_count']}")
