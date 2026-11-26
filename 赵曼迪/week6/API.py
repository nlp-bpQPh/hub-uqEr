"""
方法四·外部版：直接用大模型 API 做零样本（不训练，走网络）

教学重点：
  1. 这是**最省事的一条路**：不加载本地权重、不占显存，只要一个 key。
     冷启动验证、跨领域临时任务，它都是第一选择。
  2. 但成本和延迟都是「按次计费」的：同一批 500 条跑一遍要几十秒到几分钟，
     单条延迟是本地 0.6B 的数十倍，因此高 QPS 场景必须换成本地小模型或微调。
  3. 代码写法就是 OpenAI 兼容接口的事实标准：
          client = OpenAI(base_url=..., api_key=...)
          client.chat.completions.create(model=..., messages=[...])
     换任何 OpenAI 兼容服务（DeepSeek / 本地 vLLM / 其他云厂商）只改环境变量，代码不动。

环境变量（只需要这一个 Key，其余有默认值）：
  LLM_API_KEY        必填，服务商的 key（DeepSeek 用 DEEPSEEK_API_KEY 也行）
  LLM_BASE_URL       默认 https://api.deepseek.com
  LLM_MODEL          默认 deepseek-v4-flash
  LLM_EXTRA_HEADERS  可选，JSON 字符串。某些网关要求额外的 header（如路由用的 provider 标识），
                     用这个传即可，代码本身不绑定任何一家网关。

用法：
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500 --workers 8

###############
关于模型设置：
思考/非思考（写法下面有）
流式/非流式:stream
temperature/top-p....
message:这里调用接口发出的话自己设置（格式下面有），但是如果不是每次都从头累加计入message传入API，会没有命中KV缓存，可接近输入上限时再截断
        如果直接在对话框中问答，模型时累加输入，输入到一定量会压缩前面的话再往下进行
设置中转站处理无法请求问题：若用户A申请调用API无法返回，会调用另一家
                          宏观上连续有多个用户调用同一家API都连接失败，到达一定值，上游将这个API冷冻，隔设定的时间后（比如30s)后发一下很短的message，如果返回成功，再控制这个API可用
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#sys.path 是 Python 导入包时的搜索路径列表
#insert(0, ...) 插到最前面，优先级最高
#比如下方from common.data import load_clean_val 等价于：CODE_ROOT/common/data.py

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table

# 可选：DeepSeek 推理模型支持关闭思考（省 token、降延迟）。其它服务商不认这个参数，默认 None。
#EXTRA_BODY = {'thinking': {'type': 'disabled'}}
EXTRA_BODY = {'thinking': {'type': 'disabled'}} if os.getenv('LLM_DISABLE_THINKING', '0') in ('1', 'true') else None
#os.getenv('LLM_DISABLE_THINKING', '0')：读取环境变量 LLM_DISABLE_THINKING，环境变量不存在 → 返回默认值 '0'（字符串）
#{'thinking': {'type': 'disabled'}} ：关闭思考

SYSTEM = ('你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，'
          '只输出类别名称，不要输出任何其他内容。' + chr(10) + '可选类别：' + '、'.join(LABELS))
#chr(10) = \n 换行，等价于下面写法：
# SYSTEM = (
#     "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
#     "只输出类别名称，不要输出任何其他内容。\n"
#     f"可选类别：{'、'.join(LABELS)}"
# )


def make_client():
    """
    构造 OpenAI 兼容客户端。

    为什么 base_url / model 都从环境变量读：
      「同一份代码跑通不同服务商」是工程能力的一部分——
      线上换模型时只改配置不改代码，这一点在面试里经常被问到。
    """
    from openai import OpenAI
    import httpx

    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") #读取环境变量的LLM_API_KEY（通用大模型key)或DEEPSEEK_API_KEY(deepseek key)
                                                                    #LLM_API_KEY，DEEPSEEK_API_KEY只是约定俗成的叫法，实际自己怎么配环境变量都可以
    if not key:
        raise SystemExit("缺少 LLM_API_KEY（或 DEEPSEEK_API_KEY）环境变量")
    extra = json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}"))     # 可选：网关路由 header

    http_client = httpx.Client(timeout=60)
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"), #环境变量 LLM_BASE_URL 存在 → 用你填的地址（网关地址），没设置 → 默认 DeepSeek 官方地址 https://api.deepseek.com
        default_headers=extra or None,
        http_client=http_client,
        #timeout=60, #单次请求最长等待 60 秒，超过 60 秒直接超时断开，防止一个请求卡死线程池。
        max_retries=2,          # 偶发 429/网络抖动自动重试，不要把整批跑挂
    )
#LLM_EXTRA_HEADERS：作用：追加自定义请求头，主要给中转网关、内部代理服务用
#os.getenv("LLM_EXTRA_HEADERS", "{}")：没有设置这个环境变量时，默认返回字符串 "{}"
#json.loads()：把 JSON 字符串转成 Python 字典
#示例设置（CMD 临时）：set LLM_EXTRA_HEADERS={"X-Gateway-Route":"deepseek-r1"},现在不管，默认空字典
#extra or None：如果 extra={}（空字典），extra or None 结果就是 None，传给 default_headers=None，SDK 不会带上额外头，更干净。
#当不用官方接口，换成第三方中转网关的时候,需要用到LLM_BASE_URL，LLM_EXTRA_HEADERS
#OpenAI(...) 是 openai v1 新版 SDK 里面的类（Class），只要服务商兼容 OpenAI 接口协议（OpenAI-compatible），全都可以用这个同一个 OpenAI() 类，里面这些参数是这个 OpenAI 类预先定义好支持的入参

def parse_label(text: str) -> int: #text: str → 参数 text 应该传字符串（参数类型注解），-> int → 函数执行完返回整数
    """从返回文本里抽类别名；抽不到返回 -1（统计解析失败率）"""
    hits = [n for n in LABELS if n in text]
    return LABELS.index(max(hits, key=len)) if hits else -1 #key=len按照长度，max(hits, key=len)：按照长度取最长的hits,LABELS.index():返回最长的值所在LABELS里的索引
#text传入的是模型的返回值


def classify_one(client, model, title, max_tokens=256):
    """单条请求：返回 (预测 id, 原始输出, 用量)。异常一律吞掉并记为失败，保证整批能跑完。"""
    try:
        r = client.chat.completions.create(      #调用客户端的聊天补全接口，就是向大模型发请求，把返回结果存到变量 `r`，这是 OpenAI 兼容接口标准写法，DeepSeek、通义千问兼容版都这么写
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": title}], #模型收到的指令
            temperature=0,          # 分类要稳定，不要随机性
            max_tokens=max_tokens,  # 推理模型会先生成思考 token，留出余量；64 太小会被截断成空回复
            extra_body=EXTRA_BODY,  # 见文件头：可选关闭推理模型的思考过程
        )
        raw = (r.choices[0].message.content or "").strip()
        #r.choices：接口返回的候选回答列表，一般只返回 1 条；[0]：取第一条；.message：模型自己生成的一条消息，角色是 assistant；.content：模型刚刚生成的回答文本
        #or ""：**保护**：如果 content 是 None（接口返回空），就替换成空字符串，防止报错
        usage = {
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0), #`prompt_tokens`：输入 token（system 提示词 + 用户 title）
            "completion_tokens": getattr(r.usage, "completion_tokens", 0), #`completion_tokens`：模型输出 token（就是 raw 里面的内容）
        }
        #getattr(对象, 属性名, 默认值)：去 `r.usage` 里面找 `prompt_tokens`
        return parse_label(raw), raw, usage
    except Exception as e:          # 网络错误 / 限流 / 超时 → 记为解析失败，不中断整批
        return -1, f"ERROR: {e}", {"prompt_tokens": 0, "completion_tokens": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0) #评估样本数量，int 类型，默认跑 500 条。
    ap.add_argument("--workers", type=int, default=8, help="并发数；太高会被限流") #并发线程数，默认 8 个,并发越高跑越快，但大模型 API 有频率限制（429 限流），所以注释提醒不要开太大
    ap.add_argument("--out", default="outputs/api_zero_shot_pred.json") #输出结果文件路径，默认保存到 `outputs/api_zero_shot_pred.json`
    args = ap.parse_args()

    model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    client = make_client()
    samples, meta = load_clean_val(args.n) #加载清洗好的验证集，只取 `args.n` 条;
    meta["api_model"] = model
    meta["base_url"] = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool: #开启**线程池**并发
        results = list(pool.map(lambda d: classify_one(client, model, d["sentence"]), samples))
        #map(函数, 可迭代对象):依次从后面这个可迭代对象（这里就是`samples`）拿出一个元素，传给前面的函数，作为函数的参数
        #pool.:多线程并发执行map，pool.map 有一个特点：**不管线程谁先跑完，返回结果的顺序和 samples 的顺序保持一致
    elapsed = round(time.time() - t0, 1)

    preds = [r[0] for r in results]
    raws = [r[1] for r in results]
    tin = sum(r[2]["prompt_tokens"] for r in results) #全部输入 token 总和（system + user）
    tout = sum(r[2]["completion_tokens"] for r in results) #全部模型输出 token 总和（思考内容 + 答案）

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report.update({
        "method": f"api_zero_shot:{model}",
        "elapsed_s": elapsed,
        "sec_per_item": round(elapsed / len(samples), 2),
        "tokens_in": tin,
        "tokens_out": tout,
        "pred_distribution": pred_distribution(preds), #统计预测值分几类
    })  #生成一些指标

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    #`zip(a, b, c)` 是 Python 内置函数：**把多个可迭代对象，按位置一一配对打包**
    # 下标0：samples[0]  preds[0]  raws[0]
    # 下标1：samples[1]  preds[1]  raws[1]
    # 下标2：samples[2]  preds[2]  raws[2]
    
    print(f"模型 {model}｜{meta['n_eval']} 条｜{elapsed}s（{report['sec_per_item']}s/条）"
          f"｜tokens in/out = {tin}/{tout}")
    print(to_markdown_table({f"API zero-shot（{model}）": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

