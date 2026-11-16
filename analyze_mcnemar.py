import os
import json

import pandas as pd
from scipy.stats import binomtest


# =========================================================
# 1. 配置
# =========================================================

ROOT = "/root/calligraphy_sci"

SEEDS = [42, 123, 2026]

PRED_DIR = os.path.join(
    ROOT,
    "results",
    "final_test"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "statistical_tests"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# 2. McNemar Exact Test
#
# b:
# Baseline正确，Ours错误
#
# c:
# Baseline错误，Ours正确
#
# H0:
# b 和 c 出现概率相同
# =========================================================

results = []


for seed in SEEDS:

    baseline_path = os.path.join(
        PRED_DIR,
        f"baseline_seed{seed}_predictions.csv"
    )

    ours_path = os.path.join(
        PRED_DIR,
        f"partial_seed{seed}_predictions.csv"
    )


    baseline = pd.read_csv(
        baseline_path
    )

    ours = pd.read_csv(
        ours_path
    )


    assert len(baseline) == len(ours)


    baseline_correct = (
        baseline["character_true"]
        ==
        baseline["character_pred"]
    )


    ours_correct = (
        ours["character_true"]
        ==
        ours["character_pred"]
    )


    # -----------------------------------------------------
    # 2x2 paired table
    # -----------------------------------------------------

    both_correct = int(
        (
            baseline_correct
            &
            ours_correct
        ).sum()
    )


    baseline_correct_ours_wrong = int(
        (
            baseline_correct
            &
            ~ours_correct
        ).sum()
    )


    baseline_wrong_ours_correct = int(
        (
            ~baseline_correct
            &
            ours_correct
        ).sum()
    )


    both_wrong = int(
        (
            ~baseline_correct
            &
            ~ours_correct
        ).sum()
    )


    b = baseline_correct_ours_wrong

    c = baseline_wrong_ours_correct

    discordant = b + c


    # =====================================================
    # Exact McNemar
    #
    # 等价于对 discordant pairs 做
    # Binomial(n, 0.5) 双侧检验
    # =====================================================

    test_result = binomtest(
        k=min(b, c),
        n=discordant,
        p=0.5,
        alternative="two-sided"
    )


    p_value = float(
        test_result.pvalue
    )


    net_gain = (
        c - b
    )


    discordant_ratio = (
        c / b
        if b > 0
        else None
    )


    significant_005 = (
        p_value < 0.05
    )


    result = {

        "seed":
            seed,

        "num_samples":
            len(baseline),

        "both_correct":
            both_correct,

        "baseline_correct_ours_wrong":
            b,

        "baseline_wrong_ours_correct":
            c,

        "both_wrong":
            both_wrong,

        "discordant_pairs":
            discordant,

        "net_corrected_samples":
            net_gain,

        "ours_to_baseline_discordant_ratio":
            discordant_ratio,

        "exact_mcnemar_p_value":
            p_value,

        "significant_at_0.05":
            significant_005
    }


    results.append(
        result
    )


# =========================================================
# 3. 保存 CSV
# =========================================================

result_df = pd.DataFrame(
    results
)


csv_path = os.path.join(
    OUTPUT_DIR,
    "mcnemar_test_results.csv"
)


result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 4. 保存 JSON
# =========================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "mcnemar_test_results.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# 5. 打印论文可用结果
# =========================================================

print()

print(
    "=============================================="
)

print(
    "EXACT McNEMAR TEST"
)

print(
    "=============================================="
)


for item in results:

    print()

    print(
        f"Seed = {item['seed']}"
    )

    print(
        "Baseline correct -> Ours wrong:",
        item[
            "baseline_correct_ours_wrong"
        ]
    )

    print(
        "Baseline wrong -> Ours correct:",
        item[
            "baseline_wrong_ours_correct"
        ]
    )

    print(
        "Net corrected samples:",
        item[
            "net_corrected_samples"
        ]
    )

    print(
        "Discordant pair ratio "
        "(Ours improvement / Ours degradation):",
        f"{item['ours_to_baseline_discordant_ratio']:.4f}"
    )

    print(
        "Exact McNemar p-value:",
        f"{item['exact_mcnemar_p_value']:.8f}"
    )

    print(
        "Significant at p < 0.05:",
        item[
            "significant_at_0.05"
        ]
    )


print()

print(
    "CSV saved to:",
    csv_path
)

print(
    "JSON saved to:",
    json_path
)
