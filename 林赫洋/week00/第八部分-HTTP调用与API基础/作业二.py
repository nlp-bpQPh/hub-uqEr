import requests

BASE_URL = "https://jsonplaceholder.typicode.com"
TIMEOUT = 10


def fetch_posts():
    try:
        response = requests.get(f"{BASE_URL}/posts", timeout=TIMEOUT)
        print(f"响应状态码: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        print(f"网络请求出错: {error}")
    except ValueError:
        print("错误：响应内容不是合法的 JSON。")
    return None


def count_posts(posts):
    print()
    print(f"【帖子总数】{len(posts)}")


def filter_posts_by_user(posts, user_id=1):
    filtered = [post for post in posts if post.get("userId") == user_id]

    print()
    print(f"【用户 {user_id} 的帖子】共 {len(filtered)} 条")
    for post in filtered:
        print(f"  - [id={post['id']}] {post['title']}")


def show_first_titles(posts, count=5):
    print()
    print(f"【前 {count} 个帖子的标题】")
    for index, post in enumerate(posts[:count], start=1):
        print(f"  {index}. {post['title']}")


def find_longest_title(posts):
    if not posts:
        print("\n没有帖子数据，无法比较标题长度。")
        return

    longest = max(posts, key=lambda post: len(post.get("title", "")))

    print()
    print("【标题最长的帖子】")
    print(f"  帖子 ID : {longest['id']}")
    print(f"  用户 ID : {longest['userId']}")
    print(f"  标题长度: {len(longest['title'])}")
    print(f"  标题    : {longest['title']}")


def main():
    print("=" * 50)
    print("获取所有帖子数据")
    print("=" * 50)

    posts = fetch_posts()
    if not posts:
        print("未能获取帖子数据，程序结束。")
        return

    count_posts(posts)
    filter_posts_by_user(posts, user_id=1)
    show_first_titles(posts, count=5)
    find_longest_title(posts)


if __name__ == "__main__":
    main()
