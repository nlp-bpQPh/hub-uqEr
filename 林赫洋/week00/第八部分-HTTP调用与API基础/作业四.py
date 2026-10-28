import requests

BASE_URL = "https://jsonplaceholder.typicode.com"
TIMEOUT = 10
TITLE_LIMIT = 50


def request_json(path, params=None):
    try:
        response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("\n[错误] 请求超时，请检查网络连接。")
    except requests.exceptions.ConnectionError:
        print("\n[错误] 无法连接到服务器，请检查网络。")
    except requests.exceptions.HTTPError as error:
        print(f"\n[错误] 接口返回异常状态码: {error}")
    except ValueError:
        print("\n[错误] 响应内容不是合法的 JSON。")
    return None


def read_user_id():
    raw = input("请输入用户 ID（正整数）: ").strip()
    if not raw.isdigit() or int(raw) <= 0:
        print("\n[提示] 用户 ID 必须是正整数。")
        return None
    return int(raw)


def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def list_users():
    print_header("所有用户列表")

    users = request_json("/users")
    if users is None:
        return

    print(f"共 {len(users)} 个用户\n")
    print(f"{'ID':<4}{'姓名':<22}{'邮箱'}")
    print("-" * 60)
    for user in users:
        print(f"{user['id']:<4}{user['name']:<22}{user['email']}")


def show_one_user():
    print_header("查询指定用户")

    user_id = read_user_id()
    if user_id is None:
        return

    user = request_json(f"/users/{user_id}")
    if not user:
        return

    address = user.get("address", {})
    company = user.get("company", {})

    print(f"  ID      : {user.get('id')}")
    print(f"  姓名    : {user.get('name')}")
    print(f"  用户名  : {user.get('username')}")
    print(f"  邮箱    : {user.get('email')}")
    print(f"  电话    : {user.get('phone')}")
    print(f"  网站    : {user.get('website')}")
    print(f"  公司    : {company.get('name')}")
    print(f"  城市    : {address.get('city')}")
    print(f"  街道    : {address.get('street')}")


def list_posts():
    print_header("所有帖子")

    posts = request_json("/posts")
    if posts is None:
        return

    print(f"共 {len(posts)} 条帖子\n")
    for post in posts:
        title = post.get("title", "")
        if len(title) > TITLE_LIMIT:
            title = title[:TITLE_LIMIT] + "..."
        print(f"[{post['id']:>3}] 用户{post['userId']} | {title}")


def list_posts_by_user():
    print_header("查询指定用户的帖子")

    user_id = read_user_id()
    if user_id is None:
        return

    # 用查询参数让服务端过滤，等效于 /posts?userId=1
    posts = request_json("/posts", params={"userId": user_id})
    if not posts:
        print(f"\n[提示] 用户 {user_id} 没有帖子（或该用户不存在）。")
        return

    print(f"用户 {user_id} 共发布 {len(posts)} 条帖子：\n")
    for index, post in enumerate(posts, start=1):
        print(f"  {index}. [{post['id']}] {post['title']}")

        body = post.get("body", "").replace("\n", " ")
        if len(body) > 60:
            body = body[:60] + "..."
        print(f"     {body}\n")


def show_menu():
    print()
    print("=" * 60)
    print("JSONPlaceholder API 客户端")
    print("=" * 60)
    print("  1. 获取所有用户")
    print("  2. 获取指定用户信息")
    print("  3. 获取所有帖子")
    print("  4. 获取指定用户的帖子")
    print("  5. 退出")
    print("=" * 60)


def main():
    while True:
        show_menu()
        choice = input("请选择功能（1-5）: ").strip()

        if choice == "1":
            list_users()
        elif choice == "2":
            show_one_user()
        elif choice == "3":
            list_posts()
        elif choice == "4":
            list_posts_by_user()
        elif choice == "5":
            print("\n再见！")
            break
        else:
            print("\n[提示] 无效的选项，请输入 1 到 5 之间的数字。")


if __name__ == "__main__":
    main()
