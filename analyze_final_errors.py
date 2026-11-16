import os
import json

import numpy as np
import pandas as pd


# =========================================================
# 1. 配置
# =========================================================

SEEDS = [42, 123, 2026]

ROOT = "/root/calligraphy_sci"

TRAIN_CSV = os.path.join(
    ROOT,
    "dual_splits",
    "dual_train.csv"
)

TEST_CSV = os.path.join(
    ROOT,
    "dual_splits",
    "dual_test.csv"
)

PRED_DIR = os.path.join(
    ROOT,
    "results",
    "final_test"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_error_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# 2. 读取 Train / Test
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

test_df = pd.read_csv(
    TEST_CSV
).reset_index(drop=True)


print(
    "Train samples:",
    len(train_df)
)

print(
    "Test samples:",
    len(test_df)
)


# =========================================================
# 3. 每个字符的训练样本数量
# =========================================================

train_counts = (

    train_df
    .groupby("character_label")
    .size()
    .to_dict()
)


test_df["train_count"] = (

    test_df["character_label"]
    .map(train_counts)
)


# =========================================================
# 4. 样本数量区间
# =========================================================

def get_frequency_bin(
    count
):

    if count <= 5:
        return "3-5"

    elif count <= 10:
        return "6-10"

    elif count <= 20:
        return "11-20"

    elif count <= 30:
        return "21-30"

    else:
        return "31+"


test_df["frequency_bin"] = (

    test_df["train_count"]
    .apply(
        get_frequency_bin
    )
)


# =========================================================
# 5. 保存每个seed的错误比较
# =========================================================

seed_summaries = []

frequency_rows = []

all_seed_tables = {}


for seed in SEEDS:

    print()
    print(
        "========================================"
    )

    print(
        f"Analyzing Seed {seed}"
    )


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


    # =====================================================
    # 基础检查
    # =====================================================

    assert len(baseline) == len(test_df)
    assert len(ours) == len(test_df)


    # =====================================================
    # 合并
    # =====================================================

    df = test_df.copy()


    df["baseline_pred"] = (
        baseline["character_pred"]
        .values
    )


    df["ours_pred"] = (
        ours["character_pred"]
        .values
    )


    df["baseline_correct"] = (

        df["baseline_pred"]
        == df["character_label"]
    )


    df["ours_correct"] = (

        df["ours_pred"]
        == df["character_label"]
    )


    # =====================================================
    # 四种情况
    # =====================================================

    df["case"] = "both_wrong"


    df.loc[

        (
            df["baseline_correct"]
            &
            df["ours_correct"]
        ),

        "case"

    ] = "both_correct"


    df.loc[

        (
            ~df["baseline_correct"]
            &
            df["ours_correct"]
        ),

        "case"

    ] = "baseline_wrong_ours_correct"


    df.loc[

        (
            df["baseline_correct"]
            &
            ~df["ours_correct"]
        ),

        "case"

    ] = "baseline_correct_ours_wrong"


    # =====================================================
    # 统计
    # =====================================================

    both_correct = int(

        (
            df["case"]
            == "both_correct"
        ).sum()
    )


    improved = int(

        (
            df["case"]
            == "baseline_wrong_ours_correct"
        ).sum()
    )


    degraded = int(

        (
            df["case"]
            == "baseline_correct_ours_wrong"
        ).sum()
    )


    both_wrong = int(

        (
            df["case"]
            == "both_wrong"
        ).sum()
    )


    net_gain = (
        improved
        - degraded
    )


    print(
        "Both correct:",
        both_correct
    )

    print(
        "Baseline wrong -> Ours correct:",
        improved
    )

    print(
        "Baseline correct -> Ours wrong:",
        degraded
    )

    print(
        "Both wrong:",
        both_wrong
    )

    print(
        "Net corrected samples:",
        net_gain
    )


    seed_summaries.append({

        "seed":
            seed,

        "both_correct":
            both_correct,

        "baseline_wrong_ours_correct":
            improved,

        "baseline_correct_ours_wrong":
            degraded,

        "both_wrong":
            both_wrong,

        "net_corrected_samples":
            net_gain
    })


    # =====================================================
    # 保存逐样本对比
    # =====================================================

    output_path = os.path.join(

        OUTPUT_DIR,

        f"seed{seed}_sample_comparison.csv"
    )


    df.to_csv(

        output_path,

        index=False,

        encoding="utf-8-sig"
    )


    all_seed_tables[
        seed
    ] = df


    # =====================================================
    # 频率区间分析
    # =====================================================

    bin_order = [
        "3-5",
        "6-10",
        "11-20",
        "21-30",
        "31+"
    ]


    for frequency_bin in bin_order:

        subset = df[

            df["frequency_bin"]
            == frequency_bin
        ]


        if len(subset) == 0:

            continue


        baseline_acc = (

            subset[
                "baseline_correct"
            ].mean()
            * 100
        )


        ours_acc = (

            subset[
                "ours_correct"
            ].mean()
            * 100
        )


        frequency_rows.append({

            "seed":
                seed,

            "frequency_bin":
                frequency_bin,

            "test_samples":
                len(subset),

            "baseline_top1":
                baseline_acc,

            "ours_top1":
                ours_acc,

            "improvement":
                ours_acc
                - baseline_acc
        })


# =========================================================
# 6. 保存每seed摘要
# =========================================================

seed_summary_df = pd.DataFrame(
    seed_summaries
)


seed_summary_path = os.path.join(

    OUTPUT_DIR,

    "seed_error_summary.csv"
)


seed_summary_df.to_csv(

    seed_summary_path,

    index=False,

    encoding="utf-8-sig"
)


# =========================================================
# 7. 频率区间 Mean ± Std
# =========================================================

frequency_df = pd.DataFrame(
    frequency_rows
)


frequency_detail_path = os.path.join(

    OUTPUT_DIR,

    "frequency_bin_by_seed.csv"
)


frequency_df.to_csv(

    frequency_detail_path,

    index=False,

    encoding="utf-8-sig"
)


frequency_summary_rows = []


for frequency_bin in [

    "3-5",
    "6-10",
    "11-20",
    "21-30",
    "31+"

]:

    subset = frequency_df[

        frequency_df[
            "frequency_bin"
        ]
        == frequency_bin
    ]


    if len(subset) == 0:

        continue


    frequency_summary_rows.append({

        "frequency_bin":
            frequency_bin,

        "test_samples":
            int(
                subset[
                    "test_samples"
                ].iloc[0]
            ),

        "baseline_mean":
            float(
                subset[
                    "baseline_top1"
                ].mean()
            ),

        "baseline_std":
            float(
                subset[
                    "baseline_top1"
                ].std(
                    ddof=1
                )
            ),

        "ours_mean":
            float(
                subset[
                    "ours_top1"
                ].mean()
            ),

        "ours_std":
            float(
                subset[
                    "ours_top1"
                ].std(
                    ddof=1
                )
            ),

        "improvement_mean":
            float(
                subset[
                    "improvement"
                ].mean()
            )
    })


frequency_summary_df = pd.DataFrame(
    frequency_summary_rows
)


frequency_summary_path = os.path.join(

    OUTPUT_DIR,

    "frequency_bin_summary.csv"
)


frequency_summary_df.to_csv(

    frequency_summary_path,

    index=False,

    encoding="utf-8-sig"
)


# =========================================================
# 8. 三个seed一致改善 / 一致退化
# =========================================================

consistency_df = test_df.copy()


for seed in SEEDS:

    df = all_seed_tables[
        seed
    ]


    consistency_df[
        f"improved_seed{seed}"
    ] = (

        ~df["baseline_correct"]
        &
        df["ours_correct"]
    )


    consistency_df[
        f"degraded_seed{seed}"
    ] = (

        df["baseline_correct"]
        &
        ~df["ours_correct"]
    )


    consistency_df[
        f"baseline_pred_seed{seed}"
    ] = (

        df["baseline_pred"]
        .values
    )


    consistency_df[
        f"ours_pred_seed{seed}"
    ] = (

        df["ours_pred"]
        .values
    )


improved_columns = [

    f"improved_seed{seed}"

    for seed in SEEDS
]


degraded_columns = [

    f"degraded_seed{seed}"

    for seed in SEEDS
]


consistency_df[
    "improved_seed_count"
] = (

    consistency_df[
        improved_columns
    ]
    .sum(
        axis=1
    )
)


consistency_df[
    "degraded_seed_count"
] = (

    consistency_df[
        degraded_columns
    ]
    .sum(
        axis=1
    )
)


# =========================================================
# 9. 稳定改善样本
# =========================================================

consistent_improved = (

    consistency_df[
        consistency_df[
            "improved_seed_count"
        ]
        == len(SEEDS)
    ]
    .copy()
)


consistent_degraded = (

    consistency_df[
        consistency_df[
            "degraded_seed_count"
        ]
        == len(SEEDS)
    ]
    .copy()
)


# 也保存至少2/3 seed改善的样本

majority_improved = (

    consistency_df[
        consistency_df[
            "improved_seed_count"
        ]
        >= 2
    ]
    .copy()
)


majority_degraded = (

    consistency_df[
        consistency_df[
            "degraded_seed_count"
        ]
        >= 2
    ]
    .copy()
)


consistent_improved.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "consistent_improved_all3.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


consistent_degraded.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "consistent_degraded_all3.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


majority_improved.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "improved_at_least_2seeds.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


majority_degraded.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "degraded_at_least_2seeds.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


# =========================================================
# 10. JSON总摘要
# =========================================================

summary = {

    "num_test_samples":
        int(
            len(test_df)
        ),

    "seed_summaries":
        seed_summaries,

    "consistent_improved_all_3_seeds":
        int(
            len(
                consistent_improved
            )
        ),

    "consistent_degraded_all_3_seeds":
        int(
            len(
                consistent_degraded
            )
        ),

    "improved_at_least_2_seeds":
        int(
            len(
                majority_improved
            )
        ),

    "degraded_at_least_2_seeds":
        int(
            len(
                majority_degraded
            )
        )
}


summary_path = os.path.join(

    OUTPUT_DIR,

    "error_analysis_summary.json"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# 11. 最终输出
# =========================================================

print()

print(
    "=============================================="
)

print(
    "FINAL ERROR ANALYSIS"
)

print(
    "=============================================="
)


print()

print(
    seed_summary_df.to_string(
        index=False
    )
)


print()

print(
    "Frequency-bin performance:"
)

print()

print(
    frequency_summary_df.to_string(
        index=False
    )
)


print()

print(
    "Consistently improved in all 3 seeds:",
    len(
        consistent_improved
    )
)


print(
    "Consistently degraded in all 3 seeds:",
    len(
        consistent_degraded
    )
)


print(
    "Improved in >=2 seeds:",
    len(
        majority_improved
    )
)


print(
    "Degraded in >=2 seeds:",
    len(
        majority_degraded
    )
)


print()

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)
