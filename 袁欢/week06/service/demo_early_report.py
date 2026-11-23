"""
早报面板 Demo：模拟「每天几百条 URL 进来，先分流再给你看」

教学重点：
  1. 分流做完不是结束——产物要变成**动作**：哪些直接拦截、哪些转人工、哪些放行。
  2. 同一个服务，喂一批假数据就能演示产品形态（不用等真实接入）。
     下面 MESSAGES 全部抽自 malicious_phish.csv 的真实样本（各档混合），
     另加两条能被规则快筛直接命中的硬信号（裸 IP / host 含 @）。
  3. 注意「review」这一档：低置信的结果不会被放行，而是转人工复核，
     宁可多一步，也不要放过不确定的恶意。

用法：
  python service/demo_early_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.classify_news import ShuntService

# 模拟一早晨收到的 URL：benign 大站 + phishing 仿冒 + malware 下载 + defacement 被黑站
MESSAGES = [
    # benign
    "mp3raid.com/music/krizz_kaliko.html",
    "bopsecrets.org/rexroth/cr/1.htm",
    "http://www.pashminaonline.com/pure-pashminas",
    # phishing（含两条硬信号：host 含 @ / 裸 IP，规则快筛直接命中）
    "http://secure-banking.example.com@evil.com/webscr/login",
    "http://192.168.203.111/verify/account/login.php",
    "br-icloud.com.br",
    "signin.eby.de.zukruygxctzmmqi.civpro.co.za",
    # malware
    "http://www.824555.com/app/member/SportOption.php?uid=guest&langx=gb",
    "http://9779.info/%E5%84%BF%E7%AB%A5%E7%AB%8B%E4%BD%93%E7%BA%B8%E8%B4%B4%E7%94%BB/",
    # defacement
    "http://www.garage-pirenne.be/index.php?option=com_content&view=article&id=70&vsig70_0=15",
    "http://adventure-nicaragua.net/index.php?option=com_mailto&tmpl=component&link=aHR0cDovL2FkdmVudHVyZS1uaWNhcmFndWEubmV0L2luZGV4LnBocA",
]


def main():
    svc = ShuntService()
    rows = [svc.classify(m) for m in MESSAGES]

    buckets = {"block": [], "review": [], "pass": []}
    for r in rows:
        buckets[r["queue"]].append(r)

    def show(title, items, note):
        print(f"\n{'='*58}\n{title}（{len(items)} 条）　{note}\n{'='*58}")
        for r in items:
            print(f"  [{r['topic']}/{r['sub_topic']:<10}] conf={r['conf']:.2f} "
                  f"via {r['model']:<8} | {r['title'][:44]}")
        if not items:
            print("  （空）")

    show("① 直接拦截 block", buckets["block"], "高置信 + 恶意 → 直接拦下")
    show("② 待复核 review", buckets["review"], "置信度不足 → 转人工复核，宁多勿漏")
    show("③ 放行 pass", buckets["pass"], "良性 → 不进安全视野")

    total = len(rows)
    print(f"\n合计 {total} 条：block {len(buckets['block'])} / "
          f"review {len(buckets['review'])} / pass {len(buckets['pass'])}")
    print("规则命中：", ", ".join(r["title"][:44] for r in rows if r["model"] == "rule") or "无")


if __name__ == "__main__":
    main()
