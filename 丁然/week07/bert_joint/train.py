"""训练 BERT+Linear 或 BERT+CRF 事件抽取模型。"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TRAIN_PATH,
    get_device,
)
from data import (  # noqa: E402
    EventDataset,
    collate_examples,
    event_type_from_labels,
    extract_token_spans,
    load_rows,
    print_data_examples,
    split_rows,
)
from labels import LabelSpace  # noqa: E402
from model import EventTagger  # noqa: E402


transformers_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    # 同步 Python、PyTorch 和设备随机源，保证两个模型切分一致。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def freeze_lower_layers(model: EventTagger, layer_count: int) -> None:
    if layer_count <= 0:
        return
    # 冻结词嵌入和底层编码器，减少 CPU 环境下的反向计算量。
    for parameter in model.encoder.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.encoder.layer[:layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def _prf(
    predicted: list[set[tuple[Any, ...]]],
    gold: list[set[tuple[Any, ...]]],
) -> tuple[float, float, float]:
    # 均为 span 集合：角色和原文字符起止完全一致才算命中。
    true_positive = sum(len(p & g) for p, g in zip(predicted, gold))
    false_positive = sum(len(p - g) for p, g in zip(predicted, gold))
    false_negative = sum(len(g - p) for p, g in zip(predicted, gold))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
    )
    return precision, recall, f1


def compute_metrics(
    records: list[dict[str, Any]],
    event_types: list[str],
) -> dict[str, float]:
    """计算事件类型、触发词、论元和完整事件指标。"""

    # 先拆出事件类型，便于单独计算准确率和宏平均 F1。
    type_pred = [record["type_pred"] for record in records]
    type_gold = [record["type_gold"] for record in records]
    event_type_accuracy = sum(
        predicted == gold
        for predicted, gold in zip(type_pred, type_gold)
    ) / max(len(records), 1)

    # 逐类计算 F1 后取宏平均，避免大类掩盖“被约谈”等小类。
    f1_values: list[float] = []
    for event_type in event_types:
        tp = sum(
            predicted == event_type and gold == event_type
            for predicted, gold in zip(type_pred, type_gold)
        )
        fp = sum(
            predicted == event_type and gold != event_type
            for predicted, gold in zip(type_pred, type_gold)
        )
        fn = sum(
            predicted != event_type and gold == event_type
            for predicted, gold in zip(type_pred, type_gold)
        )
        f1_values.append(2 * tp / max(2 * tp + fp + fn, 1))

    # 触发词和论元都按完全一致的 token 起止位置计算。
    trigger_p, trigger_r, trigger_f1 = _prf(
        [record["trigger_pred"] for record in records],
        [record["trigger_gold"] for record in records],
    )
    arg_p, arg_r, arg_f1 = _prf(
        [record["arg_pred"] for record in records],
        [record["arg_gold"] for record in records],
    )
    # 样本级精确匹配要求该样本的全部同类输出完全一致。
    trigger_exact_match = sum(
        record["trigger_pred"] == record["trigger_gold"]
        for record in records
    ) / max(len(records), 1)
    arg_exact_match = sum(
        record["arg_pred"] == record["arg_gold"]
        for record in records
    ) / max(len(records), 1)
    # 严格匹配要求类型、触发词和全部论元同时正确。
    exact_match = sum(
        record["type_pred"] == record["type_gold"]
        and record["trigger_pred"] == record["trigger_gold"]
        and record["arg_pred"] == record["arg_gold"]
        for record in records
    ) / max(len(records), 1)
    # selection_score 用于保存验证集最佳权重，不替代单项指标。
    return {
        "event_type_accuracy": event_type_accuracy,
        # 保留旧字段，兼容已有两段式结果。
        "type_accuracy": event_type_accuracy,
        "type_macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "trigger_exact_match": trigger_exact_match,
        "trigger_precision": trigger_p,
        "trigger_recall": trigger_r,
        "trigger_f1": trigger_f1,
        "arg_exact_match": arg_exact_match,
        "arg_precision": arg_p,
        "arg_recall": arg_r,
        "arg_f1": arg_f1,
        "event_exact_match": exact_match,
        "selection_score": (
            0.3 * sum(f1_values) / max(len(f1_values), 1)
            + 0.3 * trigger_f1
            + 0.4 * arg_f1
        ),
    }


def build_records(
    labels: torch.Tensor,
    predictions: list[list[int]],
    label_space: LabelSpace,
    offsets_batch: list[list[tuple[int, int]]],
    rows: list[dict[str, Any]],
    row_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    """把预测和原始标注统一成原文字符级 span 后进行比较。"""

    records: list[dict[str, Any]] = []
    for row_index in range(labels.shape[0]):
        # 事件类型从正文中的“事件类型-触发词”标签提取。
        pred_tags = predictions[row_index]
        source_row = rows[int(row_indices[row_index])]
        offsets = offsets_batch[row_index]
        gold_type = source_row["event"]["type"]
        pred_type = (
            event_type_from_labels(pred_tags, label_space)
            or "无事件"
        )

        # 真实位置直接读取原始标注，避免重叠 BIO 压缩造成评估偏差。
        trigger_gold = {
            (
                "trigger",
                None,
                int(source_row["event"]["trigger"]["start"]),
                int(source_row["event"]["trigger"]["end"]),
            )
        }
        arg_gold = {
            (
                "arg",
                argument["role"],
                int(argument["start"]),
                int(argument["end"]),
            )
            for argument in source_row["event"]["args"]
        }

        # 预测先解码 token 区间，再映射回原文字符区间。
        pred_spans = extract_token_spans(
            pred_tags,
            label_space,
            pred_type,
        )
        trigger_pred = {
            (
                span.kind,
                span.role,
                offsets[span.start][0],
                offsets[span.end - 1][1],
            )
            for span in pred_spans
            if span.kind == "trigger"
        }
        arg_pred = {
            (
                span.kind,
                span.role,
                offsets[span.start][0],
                offsets[span.end - 1][1],
            )
            for span in pred_spans
            if span.kind == "arg"
        }
        # 每个样本保存一组预测和标注，最后跨样本汇总 TP/FP/FN。
        records.append(
            {
                "type_gold": gold_type,
                "type_pred": pred_type,
                "trigger_gold": trigger_gold,
                "trigger_pred": trigger_pred,
                "arg_gold": arg_gold,
                "arg_pred": arg_pred,
            }
        )
    return records


def run_epoch(
    model: EventTagger,
    loader: DataLoader,
    device: torch.device,
    label_space: LabelSpace,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    # 传入优化器表示训练，否则进入验证模式。
    training = optimizer is not None
    # 同时切换 BERT、Dropout 和 CRF 的训练状态。
    model.train(training)
    # 保存整个 epoch 的样本级结果，最后统一汇总指标。
    all_records: list[dict[str, Any]] = []
    total_loss = 0.0
    total_batches = 0
    start_time = time.time()
    # 验证阶段关闭梯度计算。
    context = torch.enable_grad() if training else torch.inference_mode()
    # row_indices 指向当前数据集自己的原始行。
    rows = loader.dataset.rows

    with context:
        for step, batch in enumerate(loader, start=1):
            # 动态 padding 后的张量统一移动到训练设备。
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # BERT 输出每个位置对应的 149 类发射分数。
            output = model(input_ids, attention_mask)
            # linear 使用交叉熵，crf 使用条件随机场负对数似然。
            loss = model.loss(
                output.emissions,
                labels,
                attention_mask,
            )

            if training:
                # 清空上一步骤残留的梯度。
                optimizer.zero_grad(set_to_none=True)
                # 根据当前 batch 的损失反向传播。
                loss.backward()
                # 梯度裁剪防止 CRF 和 BERT 联合训练时出现梯度爆炸。
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                # 使用 AdamW 更新 BERT、输出头和 CRF 参数。
                optimizer.step()

            # 预测结果仍使用原始 token 下标，之后与标签 span 对齐。
            predictions = model.decode(
                output.emissions.detach(),
                attention_mask,
                input_ids,
            )
            # 将预测标签和真实标签转换成类型、触发词和论元集合。
            all_records.extend(
                build_records(
                    labels.detach().cpu(),
                    predictions,
                    label_space,
                    batch["offsets"],
                    rows,
                    batch["row_indices"],
                )
            )
            # 累加批平均损失，供日志实时显示。
            total_loss += float(loss.detach().cpu())
            total_batches += 1

            # 定期打印，防止长时间训练时看不到进度。
            if step == 1 or step % 10 == 0 or step == len(loader):
                elapsed = time.time() - start_time
                phase = "训练" if training else "验证"
                print(
                    f"[{phase}] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={total_loss / total_batches:.4f} "
                    f"用时={elapsed:.1f}s",
                    flush=True,
                )

    # epoch 结束后统一计算所有指标。
    metrics = compute_metrics(all_records, label_space.event_types)
    # 记录当前 epoch 的平均损失。
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics, all_records


def save_outputs(
    output_dir: Path,
    model: EventTagger,
    tokenizer: object,
    label_space: LabelSpace,
    args: argparse.Namespace,
    metrics: dict[str, float],
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    # 权重、tokenizer 和标签映射分开保存，预测时可独立恢复。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存 tokenizer，保证推理时编码方式与训练一致。
    tokenizer.save_pretrained(output_dir)
    # 保存 BERT 配置，推理时可直接从结果目录恢复模型结构。
    model.encoder.config.save_pretrained(output_dir)
    # 保存完整权重，包括 linear 分类头或 CRF 转移矩阵。
    torch.save(
        {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        output_dir / "model.pt",
    )

    # 元数据记录训练规模和关键超参数，便于实验对比。
    metadata = {
        "decoder": args.decoder,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "freeze_layers": args.freeze_layers,
        "num_labels": label_space.num_labels,
        "train_size": args.train_size,
        "val_size": args.val_size,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 固定保存 149 个标签的顺序，避免推理端重新推导。
    (output_dir / "label_space.json").write_text(
        json.dumps(label_space.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # run_all.py 和后续两段式对比都读取这个指标文件。
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 验证集预测逐条保存，方便检查具体错误案例。
    with (output_dir / "predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for row, record in zip(rows, records):
            # set 不能直接写入 JSON，先转换成排序后的列表。
            serializable_record = {
                key: (
                    sorted(
                        value,
                        key=lambda item: json.dumps(
                            item,
                            ensure_ascii=False,
                        ),
                    )
                    if isinstance(value, set)
                    else value
                )
                for key, value in record.items()
            }
            item = {
                "id": row.get("id"),
                "sent_id": row.get("sent_id"),
                "text": row["text"],
                **serializable_record,
            }
            # 每条验证样本写入一行，方便检查具体错误案例。
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    # 命令行保持显式，方便 run_all.py 复用全部参数。
    parser = argparse.ArgumentParser(description="BERT 事件联合抽取")
    parser.add_argument(
        "--decoder",
        choices=["linear", "crf"],
        default="linear",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--data-debug-count",
        type=int,
        default=3,
        help="训练和验证阶段各打印多少条数据构造样例，0 表示关闭",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用 300/120 条样本快速自检",
    )
    return parser


def main() -> None:
    # 1. 读取命令行参数。
    args = build_parser().parse_args()

    # 2. smoke 模式只覆盖没有显式传入的参数。
    if args.smoke:
        args.max_train = args.max_train or 300
        args.max_val = args.max_val or 120
        args.max_length = args.max_length or 256

    # 3. 填充训练配置的默认值。
    args.max_length = args.max_length or DEFAULT_MAX_LENGTH
    args.batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    args.epochs = args.epochs or DEFAULT_EPOCHS

    # 4. linear 和 crf 默认写入不同目录，避免覆盖模型。
    args.output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.decoder
    )

    # 5. 固定随机种子，并自动选择 CPU、CUDA 或 MPS。
    set_seed(args.seed)
    device = get_device(args.device)

    # 6. 读取 train_extract.jsonl 中的全部正样本事件。
    rows = load_rows(args.train_path)

    # 7. 按事件类型分层切分，保证验证集覆盖所有类别。
    train_rows, val_rows = split_rows(
        rows,
        args.val_ratio,
        args.seed,
        args.max_train,
        args.max_val,
    )
    args.train_size = len(train_rows)
    args.val_size = len(val_rows)

    # 8. 根据 event_schema.json 建立 149 标签空间。
    label_space = LabelSpace(args.schema_path)

    # 9. 只从本地加载 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        # 中文 BERT 通常没有独立 pad token，因此复用 unk token。
        tokenizer.pad_token = tokenizer.unk_token

    # 10. 训练集和验证集使用相同的标签空间及截断长度。
    train_dataset = EventDataset(
        train_rows,
        tokenizer,
        label_space,
        args.max_length,
    )
    val_dataset = EventDataset(
        val_rows,
        tokenizer,
        label_space,
        args.max_length,
    )

    # 打印少量真实样本，核对原文、token、BIO 标签和字符区间。
    if args.data_debug_count > 0:
        print_data_examples(
            "训练集",
            train_rows,
            train_dataset.examples,
            tokenizer,
            label_space,
            args.data_debug_count,
        )
        print_data_examples(
            "验证集",
            val_rows,
            val_dataset.examples,
            tokenizer,
            label_space,
            min(2, args.data_debug_count),
        )

    # 11. 把同一批样本动态补到本批最长长度。
    def collate(batch: list[Any]) -> dict[str, Any]:
        return collate_examples(batch, tokenizer.pad_token_id)

    # 12. 训练集打乱，验证集保持固定顺序。
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # 13. 本地加载 BERT 编码器。
    encoder = AutoModel.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        dtype=torch.float32,
    )

    # 14. 创建统一输出头，并明确传入 CLS token 的编号。
    model = EventTagger(
        encoder=encoder,
        label_space=label_space,
        decoder=args.decoder,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    ).to(device)

    # 15. 冻结底部编码层，降低训练时间和内存占用。
    freeze_lower_layers(model, args.freeze_layers)

    # 16. 只把需要更新的参数交给 AdamW。
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 17. 打印训练规模，先确认数据切分和参数是否正确。
    print(
        f"decoder={args.decoder} device={device} "
        f"train={len(train_rows)} val={len(val_rows)} "
        f"labels={label_space.num_labels} "
        f"max_length={args.max_length} epochs={args.epochs} "
        f"freeze_layers={args.freeze_layers}",
        flush=True,
    )

    # 18. 初始化验证集最佳结果。
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_records: list[dict[str, Any]] | None = None

    # 19. 按 epoch 交替执行训练和验证。
    for epoch in range(1, args.epochs + 1):
        # 19.1 训练一个 epoch，并更新模型参数。
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            label_space,
            optimizer=optimizer,
            epoch=epoch,
        )
        # 19.2 在验证集上只做前向计算。
        val_metrics, val_records = run_epoch(
            model,
            val_loader,
            device,
            label_space,
            epoch=epoch,
        )
        # 19.3 打印类型、触发词、论元和完整事件指标。
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"type_acc={val_metrics['event_type_accuracy']:.4f} "
            f"type_macro_f1={val_metrics['type_macro_f1']:.4f} "
            f"trigger_em={val_metrics['trigger_exact_match']:.4f} "
            f"trigger_f1={val_metrics['trigger_f1']:.4f} "
            f"arg_em={val_metrics['arg_exact_match']:.4f} "
            f"arg_f1={val_metrics['arg_f1']:.4f} "
            f"event_em={val_metrics['event_exact_match']:.4f}",
            flush=True,
        )
        # 19.4 当前结果更好时，把权重复制到 CPU 内存。
        # 综合类型、触发词和论元指标选择验证集最佳权重。
        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = val_metrics
            best_records = val_records

    # 20. 训练结束后恢复验证集表现最好的权重。
    assert best_state is not None
    assert best_metrics is not None
    assert best_records is not None
    model.load_state_dict(best_state)
    model.cpu()

    # 21. 保存权重、tokenizer、指标和验证集预测。
    save_outputs(
        args.output_dir,
        model,
        tokenizer,
        label_space,
        args,
        best_metrics,
        best_records,
        val_rows,
    )
    # 22. 输出结果目录，方便后续预测和实验对比。
    print(f"训练结果已保存: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

