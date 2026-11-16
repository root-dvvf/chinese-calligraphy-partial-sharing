import os
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = "/root/calligraphy_sci"

TEST_CSV = f"{ROOT}/dual_splits/dual_test.csv"

NEW_DIR = f"{ROOT}/results/ablation_2x2_test"
OLD_DIR = f"{ROOT}/results/final_test"

OUT_DIR = f"{ROOT}/results/ablation_2x2_statistics"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 2026]

# ======================================================
# 1. 加载原始 Test 标签
# ======================================================

test = pd.read_csv(TEST_CSV).reset_index(drop=True)
test["index"] = np.arange(len(test))

N = len(test)

assert N == 13804, f"Test 样本数不一致: {N}"

# ======================================================
# 2. 定义实验配对
# ======================================================

PAIRS = [
    {
        "lambda": 1.0,
        "name": "equal_weight",
        "full": (
            NEW_DIR,
            "equal_fullshare_seed"
        ),
        "partial": (
            NEW_DIR,
            "equal_partial_seed"
        )
    },
    {
        "lambda": 0.1,
        "name": "fixed_weight",
        "full": (
            NEW_DIR,
            "fixed_fullshare_seed"
        ),
        "partial": (
            OLD_DIR,
            "partial_seed"
        )
    }
]


# ======================================================
# 3. 自动识别列名
# ======================================================

def find_column(df, candidates, fallback=None):

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    if fallback is not None:
        for col in df.columns:
            if fallback(str(col).lower()):
                return col

    return None


def load_and_align(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != N:
        raise ValueError(
            f"样本数错误: {path}, {len(df)}"
        )

    # 优先用测试集 index 对齐
    index_col = find_column(
        df,
        [
            "index",
            "idx",
            "sample_index",
            "sample_idx",
            "test_index",
            "test_idx"
        ]
    )

    if index_col is not None:

        indices = pd.to_numeric(
            df[index_col],
            errors="raise"
        ).astype(int)

        if (
            indices.nunique() != N
            or set(indices) != set(range(N))
        ):
            raise ValueError(
                f"index 不完整: {path}"
            )

        df = df.assign(
            _sample_index=indices
        )

    else:

        # 如果没有 index，则用 image_id 或 path
        # 进行一对一对齐，不凭行顺序猜测。
        aligned = False

        for key in ["image_id", "path"]:

            key_col = find_column(df, [key])

            if (
                key_col is None
                or key not in test.columns
            ):
                continue

            mapping = test[["index", key]]

            if mapping[key].duplicated().any():
                continue

            df = df.merge(
                mapping,
                left_on=key_col,
                right_on=key,
                how="left",
                validate="one_to_one"
            )

            if df["index"].isna().any():
                raise ValueError(
                    f"{path} 的 {key} 无法全部匹配"
                )

            df["_sample_index"] = (
                df["index"].astype(int)
            )

            aligned = True
            break

        if not aligned:
            raise ValueError(
                f"\n无法安全对齐文件:\n{path}\n"
                f"现有列: {list(df.columns)}\n"
                "需要 index、image_id 或 path。"
                "请不要直接按行顺序做统计。"
            )

    if df["_sample_index"].duplicated().any():
        raise ValueError(
            f"存在重复样本: {path}"
        )

    df = (
        df.set_index("_sample_index")
          .sort_index()
    )

    return df


# ======================================================
# 4. 识别每个任务的预测正确性
# ======================================================

def get_correct(df, task, path):

    if task == "character":

        tokens = ["character", "char"]

        pred_candidates = [
            "character_pred",
            "char_pred",
            "pred_character",
            "pred_char",
            "character_prediction",
            "predicted_character",
            "predicted_char"
        ]

        true_candidates = [
            "character_true",
            "char_true",
            "true_character",
            "true_char",
            "character_label",
            "char_label",
            "character_gt",
            "char_gt"
        ]

        correct_candidates = [
            "character_correct",
            "char_correct",
            "correct_character",
            "correct_char"
        ]

        truth = test["character_label"].to_numpy()

    else:

        tokens = ["source", "writer", "calligrapher"]

        pred_candidates = [
            "source_pred",
            "pred_source",
            "source_prediction",
            "predicted_source",
            "writer_pred",
            "calligrapher_pred"
        ]

        true_candidates = [
            "source_true",
            "true_source",
            "source_label",
            "source_gt",
            "writer_true",
            "calligrapher_true"
        ]

        correct_candidates = [
            "source_correct",
            "correct_source",
            "writer_correct",
            "calligrapher_correct"
        ]

        truth = test["source_label"].to_numpy()

    def matches(col, keyword):
        return (
            any(t in col for t in tokens)
            and keyword in col
        )

    pred_col = find_column(
        df,
        pred_candidates,
        fallback=lambda c: matches(c, "pred")
    )

    true_col = find_column(
        df,
        true_candidates
    )

    correct_col = find_column(
        df,
        correct_candidates,
        fallback=lambda c: matches(c, "correct")
    )

    # 若预测文件含真实标签，必须与原始 Test 一致
    if true_col is not None:

        recorded_truth = pd.to_numeric(
            df[true_col],
            errors="raise"
        ).to_numpy()

        if not np.array_equal(recorded_truth, truth):
            raise ValueError(
                f"{path} 的 {task} 真实标签不一致"
            )

    if pred_col is not None:

        prediction = pd.to_numeric(
            df[pred_col],
            errors="raise"
        ).to_numpy()

        correct = prediction == truth

    elif correct_col is not None:

        values = pd.to_numeric(
            df[correct_col],
            errors="raise"
        ).to_numpy()

        if not np.isin(values, [0, 1]).all():
            raise ValueError(
                f"{path} 的正确性列不是 0/1"
            )

        correct = values.astype(bool)

    else:

        raise ValueError(
            f"\n无法找到 {task} 的预测列:\n"
            f"{path}\n"
            f"现有列: {list(df.columns)}"
        )

    return correct


# ======================================================
# 5. Exact McNemar
# ======================================================

def exact_mcnemar(full_correct, partial_correct):

    both_correct = int(
        np.sum(full_correct & partial_correct)
    )

    b = int(
        np.sum(full_correct & ~partial_correct)
    )

    c = int(
        np.sum(~full_correct & partial_correct)
    )

    both_wrong = int(
        np.sum(~full_correct & ~partial_correct)
    )

    discordant = b + c

    if discordant == 0:
        p = 1.0
    else:
        p = binomtest(
            c,
            discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return {
        "both_correct": both_correct,
        "full_correct_partial_wrong": b,
        "full_wrong_partial_correct": c,
        "both_wrong": both_wrong,
        "net_corrected": c - b,
        "exact_p_value": float(p)
    }


# ======================================================
# 6. 执行全部 12 项配对检验
# ======================================================

rows = []

for pair in PAIRS:

    for seed in SEEDS:

        full_path = os.path.join(
            pair["full"][0],
            f"{pair['full'][1]}{seed}_predictions.csv"
        )

        partial_path = os.path.join(
            pair["partial"][0],
            f"{pair['partial'][1]}{seed}_predictions.csv"
        )

        print()
        print("=" * 55)
        print(pair["name"], "Seed:", seed)
        print("=" * 55)

        full = load_and_align(full_path)
        partial = load_and_align(partial_path)

        if not full.index.equals(partial.index):
            raise ValueError(
                "两个模型的样本索引未对齐"
            )

        for task in ["character", "source"]:

            full_correct = get_correct(
                full, task, full_path
            )

            partial_correct = get_correct(
                partial, task, partial_path
            )

            result = exact_mcnemar(
                full_correct,
                partial_correct
            )

            full_acc = float(
                np.mean(full_correct) * 100
            )

            partial_acc = float(
                np.mean(partial_correct) * 100
            )

            gain = partial_acc - full_acc

            row = {
                "experiment": pair["name"],
                "lambda": pair["lambda"],
                "seed": seed,
                "task": task,
                "full_top1": full_acc,
                "partial_top1": partial_acc,
                "improvement_pp": gain,
                **result
            }

            rows.append(row)

            print()
            print("Task:", task)
            print(f"Full Top-1: {full_acc:.4f}%")
            print(f"Partial Top-1: {partial_acc:.4f}%")
            print(f"Improvement: {gain:+.4f} pp")
            print(
                "Net Corrected:",
                result["net_corrected"]
            )
            print(
                "Exact p:",
                result["exact_p_value"]
            )


# ======================================================
# 7. Holm 校正，覆盖全部 12 项比较
# ======================================================

df = pd.DataFrame(rows)

p_values = df["exact_p_value"].to_numpy()

m = len(p_values)

order = np.argsort(p_values)

adjusted_sorted = np.maximum.accumulate(
    (m - np.arange(m)) * p_values[order]
)

adjusted_sorted = np.minimum(
    adjusted_sorted,
    1.0
)

adjusted = np.empty(m, dtype=float)
adjusted[order] = adjusted_sorted

df["holm_adjusted_p"] = adjusted

df["significant_holm_005"] = (
    df["holm_adjusted_p"] < 0.05
)


# ======================================================
# 8. 保存结果
# ======================================================

csv_path = os.path.join(
    OUT_DIR,
    "all_2x2_mcnemar_results.csv"
)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

json_path = os.path.join(
    OUT_DIR,
    "all_2x2_mcnemar_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rows,
        f,
        indent=4,
        ensure_ascii=False
    )


# ======================================================
# 9. 最终汇总
# ======================================================

print()
print("=" * 65)
print("FINAL 2x2 MCNEMAR SUMMARY")
print("=" * 65)

for name in ["equal_weight", "fixed_weight"]:

    for task in ["character", "source"]:

        sub = df[
            (df["experiment"] == name)
            & (df["task"] == task)
        ]

        gains = sub["improvement_pp"].to_numpy()

        print()
        print("Experiment:", name)
        print("Task:", task)

        print(
            f"Mean Improvement: "
            f"{gains.mean():+.4f} "
            f"± {gains.std(ddof=1):.4f} pp"
        )

        print(
            sub[[
                "seed",
                "net_corrected",
                "exact_p_value",
                "holm_adjusted_p",
                "significant_holm_005"
            ]].to_string(index=False)
        )

print()
print("CSV:", csv_path)
print("JSON:", json_path)
print()
print("ALL 2x2 MCNEMAR ANALYSIS COMPLETE")
