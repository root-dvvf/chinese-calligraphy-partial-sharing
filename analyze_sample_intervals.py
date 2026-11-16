import os
import pandas as pd
from sklearn.metrics import f1_score


# =========================================================
# 1. 文件路径
# =========================================================

TRAIN_CSV = "/root/calligraphy_sci/splits/train.csv"

BASELINE_PRED = (
    "/root/calligraphy_sci/results/"
    "resnet18_scratch_dedup/"
    "test_predictions.csv"
)

CB_PRED = (
    "/root/calligraphy_sci/results/"
    "resnet18_cb_loss/"
    "test_predictions.csv"
)

OUT_DIR = (
    "/root/calligraphy_sci/results/"
    "sample_interval_analysis"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =========================================================
# 2. 定义训练样本数量区间
# =========================================================

def get_interval(count):

    if 3 <= count <= 5:
        return "3-5"

    elif 6 <= count <= 10:
        return "6-10"

    elif 11 <= count <= 20:
        return "11-20"

    elif 21 <= count <= 30:
        return "21-30"

    else:
        return "31+"


# =========================================================
# 3. 统计每个类别训练样本数
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

class_info = pd.DataFrame({

    "label":
        class_counts.index,

    "train_count":
        class_counts.values
})

class_info["interval"] = (
    class_info["train_count"]
    .apply(get_interval)
)


print(
    "========== 类别区间统计 =========="
)

interval_order = [
    "3-5",
    "6-10",
    "11-20",
    "21-30",
    "31+"
]

for interval in interval_order:

    count = (
        class_info[
            class_info["interval"]
            == interval
        ]
        .shape[0]
    )

    print(
        f"{interval}: {count} 类"
    )

print()


# =========================================================
# 4. 读取预测结果
# =========================================================

baseline_df = pd.read_csv(
    BASELINE_PRED
)

cb_df = pd.read_csv(
    CB_PRED
)


# =========================================================
# 5. 合并训练样本信息
# =========================================================

baseline_df = baseline_df.merge(
    class_info,
    left_on="true_label",
    right_on="label",
    how="left"
)

cb_df = cb_df.merge(
    class_info,
    left_on="true_label",
    right_on="label",
    how="left"
)


# =========================================================
# 6. 计算每个区间指标
# =========================================================

def evaluate_interval(
    df,
    interval_name
):

    subset = df[
        df["interval"]
        == interval_name
    ].copy()

    if len(subset) == 0:

        return {
            "samples": 0,
            "classes": 0,
            "top1": 0.0,
            "macro_f1": 0.0
        }

    correct = (
        subset["true_label"]
        == subset["pred_label"]
    ).sum()

    top1 = (
        100.0
        * correct
        / len(subset)
    )

    macro_f1 = f1_score(
        subset["true_label"],
        subset["pred_label"],
        average="macro",
        zero_division=0
    )

    classes = (
        subset["true_label"]
        .nunique()
    )

    return {

        "samples":
            len(subset),

        "classes":
            classes,

        "top1":
            top1,

        "macro_f1":
            macro_f1
    }


# =========================================================
# 7. 汇总结果
# =========================================================

rows = []

for interval in interval_order:

    base_result = evaluate_interval(
        baseline_df,
        interval
    )

    cb_result = evaluate_interval(
        cb_df,
        interval
    )

    rows.append({

        "interval":
            interval,

        "classes":
            base_result["classes"],

        "test_samples":
            base_result["samples"],

        "baseline_top1":
            base_result["top1"],

        "cb_top1":
            cb_result["top1"],

        "top1_change":
            (
                cb_result["top1"]
                - base_result["top1"]
            ),

        "baseline_macro_f1":
            base_result["macro_f1"],

        "cb_macro_f1":
            cb_result["macro_f1"],

        "macro_f1_change":
            (
                cb_result["macro_f1"]
                - base_result["macro_f1"]
            )
    })


result_df = pd.DataFrame(
    rows
)


# =========================================================
# 8. 保存结果
# =========================================================

csv_path = os.path.join(
    OUT_DIR,
    "sample_interval_comparison.csv"
)

result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


class_path = os.path.join(
    OUT_DIR,
    "class_interval_assignment.csv"
)

class_info.to_csv(
    class_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. 输出终端结果
# =========================================================

print(
    "========== Sample Interval Result =========="
)

for _, row in result_df.iterrows():

    print()

    print(
        f"[{row['interval']}]"
    )

    print(
        "类别数:",
        int(row["classes"])
    )

    print(
        "测试样本数:",
        int(row["test_samples"])
    )

    print(
        f"Baseline Top-1: "
        f"{row['baseline_top1']:.2f}%"
    )

    print(
        f"CB Loss Top-1: "
        f"{row['cb_top1']:.2f}%"
    )

    print(
        f"Top-1变化: "
        f"{row['top1_change']:+.2f}%"
    )

    print(
        f"Baseline Macro-F1: "
        f"{row['baseline_macro_f1']:.4f}"
    )

    print(
        f"CB Loss Macro-F1: "
        f"{row['cb_macro_f1']:.4f}"
    )

    print(
        f"Macro-F1变化: "
        f"{row['macro_f1_change']:+.4f}"
    )


print()

print(
    "结果已保存:"
)

print(
    csv_path
)

print(
    class_path
)
