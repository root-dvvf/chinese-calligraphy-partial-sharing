import os
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# =====================================================
# 1. 实验配置
# =====================================================

ROOT = "/root/calligraphy_sci"

INPUT_DIR = os.path.join(
    ROOT, "results", "ablation_2x2_test"
)

OUTPUT_DIR = os.path.join(
    ROOT, "results", "ablation_2x2_statistics"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 2026]

TEST_SAMPLES = 13804

TASKS = ["character", "source"]

REQUIRED_COLUMNS = [
    "index",
    "character_true",
    "character_pred",
    "source_true",
    "source_pred"
]


# =====================================================
# 2. 读取预测结果
# =====================================================

def load_predictions(experiment, seed):

    path = os.path.join(
        INPUT_DIR,
        f"{experiment}_seed{seed}_predictions.csv"
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到预测文件: {path}"
        )

    df = pd.read_csv(path)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"{path} 缺少列 {col}"
            )

    df = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype(np.int64)

    if len(df) != TEST_SAMPLES:
        raise ValueError(
            f"样本数量错误: {path}"
        )

    if df["index"].duplicated().any():
        raise ValueError(
            f"存在重复 index: {path}"
        )

    if set(df["index"]) != set(range(TEST_SAMPLES)):
        raise ValueError(
            f"测试样本 index 不完整: {path}"
        )

    return df.set_index("index").sort_index()


# =====================================================
# 3. Exact McNemar
#
# b: Full 正确 -> Partial 错误
# c: Full 错误 -> Partial 正确
# =====================================================

def compare_predictions(full, partial, task, seed):

    true_col = f"{task}_true"
    pred_col = f"{task}_pred"

    if not full.index.equals(partial.index):
        raise ValueError(
            "两个模型的测试样本 index 不一致"
        )

    if not np.array_equal(
        full[true_col].to_numpy(),
        partial[true_col].to_numpy()
    ):
        raise ValueError(
            f"Seed {seed} 的 {task} 标签不一致"
        )

    truth = full[true_col].to_numpy()

    full_pred = full[pred_col].to_numpy()
    partial_pred = partial[pred_col].to_numpy()

    full_correct = full_pred == truth
    partial_correct = partial_pred == truth

    both_correct = int(
        (full_correct & partial_correct).sum()
    )

    b = int(
        (full_correct & ~partial_correct).sum()
    )

    c = int(
        (~full_correct & partial_correct).sum()
    )

    both_wrong = int(
        (~full_correct & ~partial_correct).sum()
    )

    if both_correct + b + c + both_wrong != TEST_SAMPLES:
        raise ValueError("配对计数总数错误")

    discordant = b + c

    # 双侧 Exact McNemar Test
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=c,
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    full_top1 = float(
        full_correct.mean() * 100
    )

    partial_top1 = float(
        partial_correct.mean() * 100
    )

    improvement = partial_top1 - full_top1

    return {
        "seed": seed,
        "task": task,
        "test_samples": TEST_SAMPLES,
        "both_correct": both_correct,
        "full_correct_partial_wrong_b": b,
        "full_wrong_partial_correct_c": c,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "net_corrected": c - b,
        "full_top1": full_top1,
        "partial_top1": partial_top1,
        "improvement_pp": improvement,
        "exact_p_value": float(p_value)
    }


# =====================================================
# 4. Holm 多重检验校正
# =====================================================

def holm_adjust(p_values):

    p = np.asarray(p_values, dtype=float)

    m = len(p)
    order = np.argsort(p)

    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * p[order]
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted, 1.0
    )

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


# =====================================================
# 5. 正式分析等权 Full vs Partial
# =====================================================

results = []

for seed in SEEDS:

    print()
    print("=" * 55)
    print(f"Analyzing Equal Weight Seed {seed}")
    print("=" * 55)

    full = load_predictions(
        "equal_fullshare", seed
    )

    partial = load_predictions(
        "equal_partial", seed
    )

    # 保存逐样本正确性对照
    paired = pd.DataFrame({
        "index": full.index.to_numpy()
    })

    for task in TASKS:

        truth = full[f"{task}_true"].to_numpy()
        pred_full = full[f"{task}_pred"].to_numpy()
        pred_partial = partial[f"{task}_pred"].to_numpy()

        paired[f"{task}_true"] = truth
        paired[f"{task}_full_pred"] = pred_full
        paired[f"{task}_partial_pred"] = pred_partial

        paired[f"{task}_full_correct"] = (
            pred_full == truth
        ).astype(int)

        paired[f"{task}_partial_correct"] = (
            pred_partial == truth
        ).astype(int)

        result = compare_predictions(
            full, partial, task, seed
        )

        results.append(result)

        print()
        print("Task:", task)
        print(
            "Full Top-1:",
            f"{result['full_top1']:.4f}%"
        )
        print(
            "Partial Top-1:",
            f"{result['partial_top1']:.4f}%"
        )
        print(
            "Improvement:",
            f"{result['improvement_pp']:+.4f} pp"
        )
        print(
            "Full correct -> Partial wrong:",
            result["full_correct_partial_wrong_b"]
        )
        print(
            "Full wrong -> Partial correct:",
            result["full_wrong_partial_correct_c"]
        )
        print(
            "Net corrected:",
            result["net_corrected"]
        )
        print(
            "Exact p-value:",
            f"{result['exact_p_value']:.10g}"
        )

    paired.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"equal_weight_seed{seed}_paired.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


# =====================================================
# 6. 保存校正结果
# =====================================================

df = pd.DataFrame(results)

# 对本次全部 6 项检验进行 Holm 校正
df["holm_adjusted_p"] = holm_adjust(
    df["exact_p_value"].to_numpy()
)

df["significant_raw_005"] = (
    df["exact_p_value"] < 0.05
)

df["significant_holm_005"] = (
    df["holm_adjusted_p"] < 0.05
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "equal_weight_mcnemar_results.csv"
)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 7. 统计三个随机种子的提升
# =====================================================

summaries = []

print()
print("=" * 60)
print("FINAL EQUAL WEIGHT MCNEMAR SUMMARY")
print("=" * 60)

for task in TASKS:

    part = df[df["task"] == task]

    gains = part["improvement_pp"].to_numpy()

    mean_gain = float(
        np.mean(gains)
    )

    std_gain = float(
        np.std(gains, ddof=1)
    )

    summary = {
        "task": task,
        "seeds": SEEDS,
        "mean_improvement_pp": mean_gain,
        "std_improvement_pp": std_gain,
        "all_seeds_improved": bool(
            np.all(gains > 0)
        )
    }

    summaries.append(summary)

    print()
    print("Task:", task)
    print(
        f"Mean Improvement: "
        f"{mean_gain:+.4f} ± {std_gain:.4f} pp"
    )

    print(
        "All Seeds Improved:",
        summary["all_seeds_improved"]
    )

    print()
    print(
        part[
            [
                "seed",
                "full_correct_partial_wrong_b",
                "full_wrong_partial_correct_c",
                "net_corrected",
                "exact_p_value",
                "holm_adjusted_p"
            ]
        ].to_string(index=False)
    )

summary_path = os.path.join(
    OUTPUT_DIR,
    "equal_weight_mcnemar_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summaries,
        f,
        indent=4,
        ensure_ascii=False
    )

print()
print("CSV saved:", csv_path)
print("JSON saved:", summary_path)
print()
print("EQUAL WEIGHT MCNEMAR ANALYSIS COMPLETE")
