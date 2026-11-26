"""兼容入口：真正的实现在项目根的 data_overview.py。

保留这个文件是为了让旧的调用路径继续可用：
    python analysis/data_overview.py
等价于
    python data_overview.py --report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_overview  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--report"]
    data_overview.main()
