# -*- coding: utf-8 -*-
# 作业三：文件操作异常处理 - 安全读写文件

import os


def read_file_safe(filename, encoding="utf-8"):
    """安全读取文件，失败返回 None"""
    f = None
    try:
        f = open(filename, "r", encoding=encoding)
        content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在")
        return None
    except PermissionError:
        print(f"错误：没有权限读取 '{filename}'")
        return None
    except UnicodeDecodeError:
        print(f"错误：'{filename}' 编码错误，无法用 {encoding} 解码")
        return None
    except IsADirectoryError:
        print(f"错误：'{filename}' 是目录，不是文件")
        return None
    except OSError as e:
        print(f"错误：读取失败，{e}")
        return None
    else:
        print(f"读取成功：'{filename}'，共 {len(content)} 个字符")
        return content
    finally:
        if f is not None and not f.closed:
            f.close()
        print(f"  read_file_safe 结束：{filename}")


def write_file_safe(filename, content, mode="w", encoding="utf-8"):
    """安全写入文件，成功返回 True，失败返回 False"""
    f = None
    try:
        f = open(filename, mode, encoding=encoding)
        f.write(content)
    except PermissionError:
        print(f"错误：没有权限写入 '{filename}'")
        return False
    except FileNotFoundError:
        print(f"错误：路径 '{filename}' 不存在")
        return False
    except IsADirectoryError:
        print(f"错误：'{filename}' 是目录，不能写入")
        return False
    except TypeError:
        print("错误：写入内容必须是字符串")
        return False
    except OSError as e:
        print(f"错误：写入失败，{e}")
        return False
    else:
        print(f"写入成功：'{filename}'")
        return True
    finally:
        if f is not None and not f.closed:
            f.close()
        print(f"  write_file_safe 结束：{filename}")


def copy_file_safe(src, dst, encoding="utf-8"):
    """安全复制文件，成功返回 True，失败返回 False"""
    try:
        content = read_file_safe(src, encoding)
        if content is None:
            raise IOError(f"源文件 '{src}' 读取失败，复制终止")
        if not write_file_safe(dst, content, "w", encoding):
            raise IOError(f"目标文件 '{dst}' 写入失败，复制终止")
    except IOError as e:
        print(f"错误：{e}")
        return False
    except Exception as e:
        print(f"错误：复制时发生异常，{e}")
        return False
    else:
        print(f"复制成功：'{src}' -> '{dst}'")
        return True
    finally:
        print("  copy_file_safe 结束")


def main():
    print("===== 作业三：安全读写文件 =====")

    test_file = "test_demo.txt"
    copy_file = "test_demo_copy.txt"
    bad_file = "test_bad.txt"

    print("\n【测试写入】")
    print(write_file_safe(test_file, "你好，Python！\n这是测试文件。\n"))
    print("写入到不存在的目录：", write_file_safe("不存在的目录/a.txt", "测试"))
    print("内容不是字符串：", write_file_safe(bad_file, 12345))

    print("\n【测试读取】")
    print(repr(read_file_safe(test_file)))
    print("读取不存在的文件：", read_file_safe("no_such_file.txt"))

    print("\n【测试复制】")
    print(copy_file_safe(test_file, copy_file))
    print(repr(read_file_safe(copy_file)))
    print("复制不存在的文件：", copy_file_safe("no_such_file.txt", "target.txt"))

    # 清理测试文件
    print("\n【清理】")
    for t in (test_file, copy_file, bad_file):
        try:
            if os.path.exists(t):
                os.remove(t)
                print(f"  已删除：{t}")
        except OSError as e:
            print(f"  删除失败：{t}，{e}")


if __name__ == "__main__":
    main()
