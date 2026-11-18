import requests

BASE_URL = "https://jsonplaceholder.typicode.com"
TIMEOUT = 10


def get_json(url):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        print(f"响应状态码: {response.status_code}")
        response.raise_for_status()  # 非 2xx 状态码抛出 HTTPError
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


def show_user_list():
    print("=" * 50)
    print("1. 获取用户列表")
    print("=" * 50)

    users = get_json(f"{BASE_URL}/users")
    if users is None:
        print("获取用户列表失败，程序结束。")
        return

    print(f"共获取到 {len(users)} 个用户，以下是前 3 个：")
    for index, user in enumerate(users[:3], start=1):
        print(f"  {index}. 姓名: {user['name']:<20} 邮箱: {user['email']}")


def show_user_detail(user_id=1):
    print()
    print("=" * 50)
    print(f"2. 获取用户 ID = {user_id} 的详细信息")
    print("=" * 50)

    user = get_json(f"{BASE_URL}/users/{user_id}")
    if user is None:
        print("获取用户信息失败。")
        return

    address = user.get("address", {})
    company = user.get("company", {})

    print(f"  姓名    : {user.get('name', '')}")
    print(f"  用户名  : {user.get('username', '')}")
    print(f"  邮箱    : {user.get('email', '')}")
    print(f"  电话    : {user.get('phone', '')}")
    print(f"  网站    : {user.get('website', '')}")
    print(f"  城市    : {address.get('city', '')}")
    print(f"  街道    : {address.get('street', '')}")
    print(f"  公司    : {company.get('name', '')}")


def main():
    show_user_list()
    show_user_detail(1)


if __name__ == "__main__":
    main()
