import requests

BASE_URL = "https://jsonplaceholder.typicode.com"
TIMEOUT = 10

new_post = {
    "title": "我的第一个 API 帖子",
    "body": "这是通过 requests.post() 提交的内容，用于练习 POST 请求。",
    "userId": 1,
}


def create_post(data):
    try:
        response = requests.post(f"{BASE_URL}/posts", json=data, timeout=TIMEOUT)
        print(f"响应状态码: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("错误：请求超时，请检查网络连接。")
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器，请检查网络。")
    except requests.exceptions.HTTPError as error:
        print(f"错误：服务器返回异常状态码 -> {error}")
    except ValueError:
        print("错误：响应内容不是合法的 JSON。")
    return None


def verify_data(sent, received):
    print()
    print("=" * 50)
    print("验证返回数据")
    print("=" * 50)

    all_matched = True
    for key, value in sent.items():
        matched = received.get(key) == value
        all_matched = all_matched and matched
        print(f"  {key:<8}: {'一致' if matched else '不一致'}")

    print()
    print("结论：返回数据与提交数据一致。" if all_matched else "结论：返回数据与提交数据存在差异。")


def main():
    print("=" * 50)
    print("创建新帖子")
    print("=" * 50)
    print("提交的数据：")
    for key, value in new_post.items():
        print(f"  {key}: {value}")

    print()
    created = create_post(new_post)
    if created is None:
        print("创建帖子失败，程序结束。")
        return

    print()
    print("服务器返回的数据：")
    for key, value in created.items():
        print(f"  {key}: {value}")

    verify_data(new_post, created)


if __name__ == "__main__":
    main()
