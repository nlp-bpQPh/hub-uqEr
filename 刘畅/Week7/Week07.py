#!/usr/bin/env python3
r"""第 7 周作业：比较金融事件抽取的模型与提示方法。

Python 3.10+，仅使用标准库和随课代码。在项目根目录运行：

    python3 Homework/compare_extraction.py --archive

配置对应的 API Key 环境变量后，可在同一批样本上比较两个模型：

    python3 Homework/compare_extraction.py --n 20 \
      --api deepseek deepseek-flash https://api.deepseek.com DEEPSEEK_API_KEY \
      --api qwen qwen-plus "$DASHSCOPE_BASE_URL" DASHSCOPE_API_KEY

DASHSCOPE_BASE_URL 填工作空间的 OpenAI 兼容 Base URL。
--api 可重复，也可在同一 API 上配置不同模型。每个模型均运行 zero-shot 和 few-shot。
--data-dir 可重复，每个目录提供 dev_extract.jsonl、train_extract.jsonl 和 event_schema.json。
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

if sys.version_info < (3, 10):
    raise SystemExit("课件的事件解析模块需要 Python 3.10 或更高版本。")

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
DATA = CODE / "data" / "processed"
ARCHIVE = CODE / "outputs"

sys.path[:0] = [str(CODE / "llm_zero_shot"), str(CODE / "common")]
from event_data import (  # noqa: E402
    SEED, load_jsonl, load_schema, make_pred_row, sample_rows, to_target_json,
    write_gold_subset,
)
from event_prompt import FEW_SHOT_EXAMPLES, build_messages, parse_model_output  # noqa: E402
from evaluate_events import as_event_list, compare_events, evaluate, row_key  # noqa: E402


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("需要正整数")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="比较多个 API 模型的金融事件抽取效果")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", action="store_true", help="离线复算课件存档（默认）")
    mode.add_argument("--api", action="append", nargs=4,
                      metavar=("别名", "模型名", "BASE_URL", "密钥环境变量"),
                      help="启用真实 API，可重复传入")
    parser.add_argument("--data-dir", type=Path, action="append",
                        help="在线评测的数据目录，可重复，默认使用 DuEE-Fin processed")
    parser.add_argument("--n", type=positive_int, help="每份数据的样本数（在线 20，离线 200）")
    parser.add_argument("--seed", type=int, default=SEED, help="在线采样种子，默认 42")
    parser.add_argument("--max-tokens", type=positive_int, default=512, help="API 输出上限")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results",
                        help="结果根目录，每次运行新建子目录")
    args = parser.parse_args()
    if not args.api and (args.data_dir or args.seed != SEED):
        parser.error("存档模式使用课件固定的 DuEE-Fin 评测子集，--data-dir 和 --seed 用于在线评测")
    return args


def api_configs(specs):
    configs = []
    aliases = set()
    for alias, model, base_url, key_env in specs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", alias) or alias in aliases:
            raise ValueError(f"API 别名必须唯一，并使用字母、数字、短横线或下划线：{alias}")
        aliases.add(alias)
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or any(
            (url.username, url.password, url.query, url.fragment)
        ):
            raise ValueError(f"{alias} 的 BASE_URL 必须是有效的 API 基础地址")
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise ValueError(f"{alias} 缺少环境变量 {key_env}")
        configs.append((alias, model, base_url.rstrip("/"), key))
    return configs


def call_api(config, messages, max_tokens):
    _, model, base_url, key = config
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if urlsplit(base_url).hostname == "api.deepseek.com":
        payload["thinking"] = {"type": "disabled"}
    request = Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                body = json.load(response)
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            raw = choice["message"]["content"]
            if not isinstance(usage, dict) or (raw is not None and not isinstance(raw, str)):
                raise TypeError("API 返回的 text 或 usage 字段类型错误")
            return {
                "raw": raw or "", "latency_s": time.perf_counter() - start,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "returned_model": body.get("model"),
            }
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
            retry = exc.code == 429 or 500 <= exc.code < 600
            exc.close()
        except (URLError, OSError, HTTPException) as exc:
            error, retry = type(exc).__name__, True
        except (ValueError, TypeError, IndexError, KeyError):
            error, retry = "API 响应格式异常", False
        if not retry or attempt == 2:
            break
        time.sleep(2 ** attempt)
    return {"raw": "", "latency_s": time.perf_counter() - start, "error": error}


def index_rows(rows, source):
    index = {}
    for row in rows:
        if not row.get("id") or not row.get("sent_id"):
            raise ValueError(f"{source} 缺少 id 或 sent_id")
        key = row_key(row)
        if key in index:
            raise ValueError(f"{source} 中样本键重复：{key}")
        index[key] = row
    return index


def prepare_dataset(directory, n, seed):
    dev = load_jsonl(directory / "dev_extract.jsonl")
    if not dev:
        raise ValueError(f"{directory} 的 dev_extract.jsonl 为空")
    dev_index = index_rows(dev, directory / "dev_extract.jsonl")
    dev_texts = {row["text"].strip() for row in dev}
    schema = load_schema(directory / "event_schema.json")
    examples = []
    if directory.resolve() == DATA:
        examples = [ex for ex in FEW_SHOT_EXAMPLES if ex["text"].strip() not in dev_texts]
    else:
        # 自定义数据集从训练集为不同事件类型各选一条，跳过与验证集重合的样本。
        seen_types, seen_texts = set(), set()
        train = load_jsonl(directory / "train_extract.jsonl")
        for row in sample_rows(train, len(train), seed):
            event = row.get("event") or {}
            event_type = event.get("type")
            text = row["text"].strip()
            if (not text or text in dev_texts or text in seen_texts
                    or row_key(row) in dev_index or event_type in seen_types
                    or event_type not in schema["event_types"]):
                continue
            examples.append({"text": row["text"], "target": to_target_json(event)})
            seen_types.add(event_type)
            seen_texts.add(text)
            if len(examples) == len(FEW_SHOT_EXAMPLES):
                break
    if not examples:
        raise ValueError(f"{directory} 没有可用于 few-shot 的独立训练示例")
    return sample_rows(dev, n, seed), schema, examples


def generate(rows, schema, examples, config, max_tokens):
    demo_messages = []
    for example in examples:
        demo_messages.extend((
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["target"]},
        ))
    for i, row in enumerate(rows, 1):
        system, user = build_messages(row["text"], schema)
        system["content"] = system["content"].replace("13 类", f"{len(schema['event_types'])} 类")
        result = call_api(config, [system, *demo_messages, user], max_tokens)
        identity = {field: row[field] for field in ("id", "sent_id", "uid") if field in row}
        yield {**identity, **result}
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)


def parse_responses(rows, raw_rows, schema):
    raw_index = index_rows(raw_rows, "模型回复")
    predictions, parse_counts = [], Counter()
    for row in rows:
        raw = raw_index[row_key(row)]["raw"]
        try:
            events, info = parse_model_output(raw, row["text"], schema)
        except TypeError:
            # 合法 JSON 也可能给出错误的字段类型，例如 arguments: 42。
            events, info = [], {"invalid_field_type": 1}
        prediction = make_pred_row(row, events)
        if "uid" in row:
            prediction["uid"] = row["uid"]
        predictions.append(prediction)
        parse_counts.update(info)
    return predictions, dict(parse_counts)


def usage_total(raw_rows, field):
    values = [row.get(field) for row in raw_rows if not row.get("error")]
    return sum(values) if values and all(isinstance(value, int) for value in values) else None


def response_stats(raw_rows, parse_counts):
    latencies = [row["latency_s"] for row in raw_rows if "latency_s" in row]
    errors = Counter(row["error"] for row in raw_rows if row.get("error"))
    return {
        "api_errors": errors,
        "api_error_count": sum(errors.values()),
        "parse_rate": round(parse_counts.get("json_ok", 0) / len(raw_rows), 4),
        "parse_counts": parse_counts,
        "truncated_count": sum(row.get("finish_reason") == "length" for row in raw_rows),
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else None,
        "prompt_tokens": usage_total(raw_rows, "prompt_tokens"),
        "completion_tokens": usage_total(raw_rows, "completion_tokens"),
        "returned_models": sorted({row["returned_model"] for row in raw_rows if row.get("returned_model")}),
    }


def badcases(gold, predictions, errors, limit=10):
    pred_index = index_rows(predictions, "预测")
    for row in gold:
        key = row_key(row)
        prediction = pred_index[key]
        g, p = as_event_list(row), as_event_list(prediction)
        trigger, argument, exact = compare_events(g, p)
        if not any(exact[1:]) and key not in errors:
            continue
        flags = []
        if key in errors:
            flags.append("API 请求失败")
        if not p:
            flags.append("漏抽事件")
        elif Counter(e["type"] for e in g) != Counter(e["type"] for e in p):
            flags.append("事件类型或数量不一致")
        else:
            if any(trigger[1:]):
                flags.append("触发词不一致")
            if any(argument[1:]):
                flags.append("论元角色或文本不一致")
        yield {"id": row["id"], "sent_id": row["sent_id"], "text": row["text"],
               "flags": flags, "gold": g, "prediction": p}
        limit -= 1
        if limit == 0:
            break


def record(folder, name, gold, predictions, meta, raw_rows=None):
    metrics = evaluate(gold, predictions, verbose=False)
    write_gold_subset(folder / f"{name}_pred.jsonl", predictions)
    errors = set()
    if raw_rows is not None:
        write_gold_subset(folder / f"{name}_raw.jsonl", raw_rows)
        errors = {row_key(row) for row in raw_rows if row.get("error")}
    write_gold_subset(folder / f"{name}_badcases.jsonl", list(badcases(gold, predictions, errors)))
    report = {**meta, **metrics}
    (folder / f"{name}_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return {
        "dataset": meta["dataset"], "source": meta["source"], "model": meta["model"],
        "method": meta["method"], "n": metrics["num_samples"],
        "trigger_f1": metrics["trigger"]["f1"], "argument_f1": metrics["argument"]["f1"],
        "event_f1": metrics["event_exact"]["f1"], "type_macro_f1": metrics["type_macro_f1"],
        **{field: meta.get(field) for field in (
            "parse_rate", "api_error_count", "latency_mean_s", "prompt_tokens", "completion_tokens",
        )},
    }


def archive_results(folder, n):
    llm = ARCHIVE / "llm"
    archived_gold = load_jsonl(llm / "gold_n200.jsonl")
    gold = archived_gold[:n]
    write_gold_subset(folder / "gold.jsonl", gold)
    base = {"source": "存档复算", "dataset": "DuEE-Fin", "seed": SEED}
    schema = load_schema()
    for method in ("zero", "few"):
        tag = f"api_{method}_n200"
        index = index_rows(load_jsonl(llm / f"raw_{tag}.jsonl"), tag)
        raw_rows = [index[row_key(row)] for row in gold]
        predictions, counts = parse_responses(gold, raw_rows, schema)
        stats = response_stats(raw_rows, counts)
        report = json.loads((llm / f"report_{tag}.json").read_text(encoding="utf-8"))
        if len(gold) == len(archived_gold):
            stats.update({key: report[key] for key in ("prompt_tokens", "completion_tokens")})
        meta = {**base, **stats, "model": report["model"], "method": method}
        yield record(folder, tag, gold, predictions, meta, raw_rows)
    for model, method, path in (
        ("词典+正则", "rule", ARCHIVE / "disc" / "pred_rule.jsonl"),
        ("BERT+CRF", "hard_constraint", ARCHIVE / "disc" / "pred_bert_crf_256_hard.jsonl"),
        ("两段式", "pipeline", ARCHIVE / "disc" / "pred_pipeline.jsonl"),
        ("Qwen3-0.6B", "lora_sft", llm / "pred_sft_lora_n200.jsonl"),
    ):
        index = index_rows(load_jsonl(path), path)
        predictions = [index[row_key(row)] for row in gold]
        yield record(folder, method, gold, predictions, {**base, "model": model, "method": method})


def online_results(folder, datasets, configs, args):
    for i, (directory, (gold, schema, examples)) in enumerate(datasets, 1):
        dataset = "DuEE-Fin" if directory.resolve() == DATA else directory.name
        target = folder / f"{i:02d}_{directory.name}"
        write_gold_subset(target / "gold.jsonl", gold)
        for config in configs:
            alias, model, base_url, _ = config
            for method, demos in (("zero", []), ("few", examples)):
                print(f"{dataset} | {alias} / {model} | {method}-shot", flush=True)
                raw_rows = list(generate(gold, schema, demos, config, args.max_tokens))
                predictions, counts = parse_responses(gold, raw_rows, schema)
                meta = {
                    "dataset": dataset, "data_dir": str(directory.resolve()), "source": "API 实测",
                    "alias": alias, "model": model, "api_host": urlsplit(base_url).hostname,
                    "method": method, "few_shot_examples": demos, "seed": args.seed,
                    "max_tokens": args.max_tokens, **response_stats(raw_rows, counts),
                }
                result = record(target, f"{alias}_{method}", gold, predictions, meta, raw_rows)
                yield {**result, "alias": alias}


def main():
    args = parse_args()
    try:
        if args.api:
            configs = api_configs(args.api)
            datasets = [(directory, prepare_dataset(directory, args.n or 20, args.seed))
                        for directory in args.data_dir or [DATA]]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="api_" if args.api else "archive_", dir=args.out_dir))
        print(f"实验结果目录 {folder.resolve()}", flush=True)
        results = list(online_results(folder, datasets, configs, args) if args.api
                       else archive_results(folder, args.n or 200))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"实验失败：{exc}") from exc

    with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("数据集 / 模型 / 方法 | 触发词 F1 | 论元 F1 | 整事件 F1 | API 失败数")
    for row in results:
        label = row.get("alias", row["model"])
        failures = row["api_error_count"]
        print(f'{row["dataset"]} / {label} / {row["method"]} | {row["trigger_f1"]:.4f} | '
              f'{row["argument_f1"]:.4f} | {row["event_f1"]:.4f} | {failures if failures is not None else "-"}')
    if args.api and any(row["api_error_count"] == row["n"] for row in results):
        raise SystemExit("至少一组 API 请求全部失败，请检查对应的 metrics.json。")


if __name__ == "__main__":
    main()
