"""轻量级分类指标计算，避免引入重依赖（numpy / sklearn）。"""

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple, Union

Label = Union[str, int, float]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_report(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    labels: Iterable[Label] = None,
) -> Tuple[float, Dict[Label, Dict[str, float]]]:
    """
    返回总体 accuracy 与逐标签的 precision/recall/f1。
    labels 不传则使用 y_true ∪ y_pred。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")

    label_set = list(labels) if labels is not None else list(
        sorted(set(y_true) | set(y_pred))
    )
    counts = {l: Counter() for l in label_set}  # tp/fp/fn 计数

    total = len(y_true)
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
            counts[t]["tp"] += 1
        else:
            counts[p]["fp"] += 1
            counts[t]["fn"] += 1

    accuracy = _safe_div(correct, total)

    per_label = {}
    for l in label_set:
        tp = counts[l]["tp"]
        fp = counts[l]["fp"]
        fn = counts[l]["fn"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_label[l] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return accuracy, per_label


def compute_metrics(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    requested: List[str],
    labels: Iterable[Label] = None,
) -> Dict[str, float]:
    """
    根据 requested 列表返回所需指标。
    支持：
      - accuracy
      - f1_score / f1_macro（同义，宏平均）
      - f1_{label}（如 f1_1, f1_pos 等，label 按字符串匹配）
      - precision_{label} / recall_{label}
    """
    accuracy, per_label = classification_report(y_true, y_pred, labels=labels)

    # 预计算宏平均
    f1_macro = 0.0
    if per_label:
        f1_macro = sum(v["f1"] for v in per_label.values()) / len(per_label)

    results: Dict[str, float] = {}
    for name in requested:
        if name == "accuracy":
            results["accuracy"] = accuracy
        elif name in ("f1_macro", "f1_score"):
            results["f1_macro"] = f1_macro
        elif name.startswith("f1_"):
            key = name[len("f1_") :]
            if key in per_label:
                results[name] = per_label[key]["f1"]
        elif name.startswith("precision_"):
            key = name[len("precision_") :]
            if key in per_label:
                results[name] = per_label[key]["precision"]
        elif name.startswith("recall_"):
            key = name[len("recall_") :]
            if key in per_label:
                results[name] = per_label[key]["recall"]
        # 其他未识别指标忽略，避免报错中断

    return results

