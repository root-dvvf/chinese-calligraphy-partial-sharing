import os
import random
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 1. 基础配置
# =========================================================

ROOT = "/root/calligraphy_sci"

TRAIN_CSV = os.path.join(ROOT, "dual_splits", "dual_train.csv")
VAL_CSV   = os.path.join(ROOT, "dual_splits", "dual_val.csv")
TEST_CSV  = os.path.join(ROOT, "dual_splits", "dual_test.csv")

OUTPUT_DIR = os.path.join(ROOT, "paper_figs_dataset_examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 是否优先显示“字符文字/来源名称”
# 如果你的CSV里没有这些列，就自动退回用 label id
POSSIBLE_CHARACTER_NAME_COLS = [
    "character_text",
    "character",
    "char_text",
    "char",
    "character_name"
]

POSSIBLE_SOURCE_NAME_COLS = [
    "source_name",
    "calligrapher",
    "calligrapher_name",
    "source",
    "writer_name"
]


# =========================================================
# 2. 读取数据
# =========================================================

def load_all_data():
    dfs = []
    for split_name, csv_path in [
        ("train", TRAIN_CSV),
        ("val", VAL_CSV),
        ("test", TEST_CSV),
    ]:
        df = pd.read_csv(csv_path)
        df["split"] = split_name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 基本检查
    required_cols = ["path", "character_label", "source_label"]
    for c in required_cols:
        if c not in df_all.columns:
            raise ValueError(f"缺少必要列: {c}")

    return df_all


# =========================================================
# 3. 自动识别可显示名称列
# =========================================================

def find_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# 4. 安全读取图像
# =========================================================

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


# =========================================================
# 5. 随机样例总览图
# =========================================================

def plot_random_overview(df, out_path, metadata_rows, nrows=3, ncols=4):
    sample_n = nrows * ncols
    df_sample = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(df_sample):
            continue

        row = df_sample.iloc[i]
        img = load_image(row["path"])

        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Load Error", ha="center", va="center")

        title = f"C{row['character_label']} | S{row['source_label']}"
        ax.set_title(title, fontsize=9)

        metadata_rows.append({
            "figure_name": "dataset_random_overview.png",
            "panel_index": i,
            "path": row["path"],
            "character_label": int(row["character_label"]),
            "source_label": int(row["source_label"]),
            "split": row["split"]
        })

    plt.suptitle("Random Dataset Overview", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 6. 同一字符，不同来源
#    每一行是一个字符；每一列是同一字符的不同来源样本
# =========================================================

def plot_same_character_diff_sources(
    df,
    out_path,
    metadata_rows,
    character_name_col=None,
    source_name_col=None,
    nrows=4,
    ncols=4
):
    # 找到至少在 ncols 个不同 source 中出现过的字符
    char_source_counts = (
        df.groupby("character_label")["source_label"]
        .nunique()
        .sort_values(ascending=False)
    )

    eligible_chars = char_source_counts[char_source_counts >= ncols].index.tolist()

    if len(eligible_chars) == 0:
        print("没有找到满足条件的字符用于‘同一字符-不同来源’图。")
        return

    # 选出现次数多的前 nrows 个字符
    char_total_counts = df[df["character_label"].isin(eligible_chars)]["character_label"].value_counts()
    selected_chars = char_total_counts.index[:nrows].tolist()

    fig, axes = plt.subplots(len(selected_chars), ncols, figsize=(3*ncols, 3*len(selected_chars)))

    if len(selected_chars) == 1:
        axes = np.array([axes])

    for r, char_label in enumerate(selected_chars):
        df_char = df[df["character_label"] == char_label].copy()

        # 选这个字符中出现次数最多的 ncols 个来源
        top_sources = (
            df_char["source_label"].value_counts().index[:ncols].tolist()
        )

        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")

            if c >= len(top_sources):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                continue

            src = top_sources[c]
            df_cell = df_char[df_char["source_label"] == src]

            row = df_cell.sample(n=1, random_state=RANDOM_SEED + r + c).iloc[0]
            img = load_image(row["path"])

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Load Error", ha="center", va="center")

            # 标题显示来源
            if source_name_col and source_name_col in row.index:
                src_text = str(row[source_name_col])
            else:
                src_text = f"S{src}"

            ax.set_title(src_text, fontsize=9)

            if c == 0:
                # 行标签显示字符
                if character_name_col and character_name_col in row.index:
                    char_text = str(row[character_name_col])
                    ax.set_ylabel(f"Char {char_text}\n(ID {char_label})", fontsize=10)
                else:
                    ax.set_ylabel(f"Char ID {char_label}", fontsize=10)

            metadata_rows.append({
                "figure_name": "dataset_same_character_diff_sources.png",
                "panel_row": r,
                "panel_col": c,
                "path": row["path"],
                "character_label": int(row["character_label"]),
                "source_label": int(row["source_label"]),
                "split": row["split"]
            })

    plt.suptitle("Examples of the Same Character Across Different Sources", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 7. 同一来源，不同字符
#    每一行是一个来源；每一列是该来源下的不同字符
# =========================================================

def plot_same_source_diff_characters(
    df,
    out_path,
    metadata_rows,
    character_name_col=None,
    source_name_col=None,
    nrows=4,
    ncols=4
):
    # 找到至少有 ncols 个不同字符的来源
    source_char_counts = (
        df.groupby("source_label")["character_label"]
        .nunique()
        .sort_values(ascending=False)
    )

    eligible_sources = source_char_counts[source_char_counts >= ncols].index.tolist()

    if len(eligible_sources) == 0:
        print("没有找到满足条件的来源用于‘同一来源-不同字符’图。")
        return

    # 选样本总数多的前 nrows 个来源
    source_total_counts = df[df["source_label"].isin(eligible_sources)]["source_label"].value_counts()
    selected_sources = source_total_counts.index[:nrows].tolist()

    fig, axes = plt.subplots(len(selected_sources), ncols, figsize=(3*ncols, 3*len(selected_sources)))

    if len(selected_sources) == 1:
        axes = np.array([axes])

    for r, source_label in enumerate(selected_sources):
        df_src = df[df["source_label"] == source_label].copy()

        # 选这个来源中出现次数最多的 ncols 个字符
        top_chars = (
            df_src["character_label"].value_counts().index[:ncols].tolist()
        )

        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")

            if c >= len(top_chars):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                continue

            char_label = top_chars[c]
            df_cell = df_src[df_src["character_label"] == char_label]

            row = df_cell.sample(n=1, random_state=RANDOM_SEED + r + c + 100).iloc[0]
            img = load_image(row["path"])

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Load Error", ha="center", va="center")

            # 标题显示字符
            if character_name_col and character_name_col in row.index:
                char_text = str(row[character_name_col])
                ax.set_title(f"{char_text}", fontsize=9)
            else:
                ax.set_title(f"C{char_label}", fontsize=9)

            if c == 0:
                if source_name_col and source_name_col in row.index:
                    src_text = str(row[source_name_col])
                    ax.set_ylabel(f"{src_text}\n(ID {source_label})", fontsize=10)
                else:
                    ax.set_ylabel(f"Source ID {source_label}", fontsize=10)

            metadata_rows.append({
                "figure_name": "dataset_same_source_diff_characters.png",
                "panel_row": r,
                "panel_col": c,
                "path": row["path"],
                "character_label": int(row["character_label"]),
                "source_label": int(row["source_label"]),
                "split": row["split"]
            })

    plt.suptitle("Examples of Different Characters Within the Same Source", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 8. 主函数
# =========================================================

def main():
    df = load_all_data()

    print("总样本数:", len(df))
    print("字符类别数:", df["character_label"].nunique())
    print("来源类别数:", df["source_label"].nunique())

    character_name_col = find_first_existing_column(df, POSSIBLE_CHARACTER_NAME_COLS)
    source_name_col = find_first_existing_column(df, POSSIBLE_SOURCE_NAME_COLS)

    print("字符显示列:", character_name_col)
    print("来源显示列:", source_name_col)

    metadata_rows = []

    # 1) 随机总览
    out1 = os.path.join(OUTPUT_DIR, "dataset_random_overview.png")
    plot_random_overview(df, out1, metadata_rows, nrows=3, ncols=4)
    print("已保存:", out1)

    # 2) 同一字符，不同来源
    out2 = os.path.join(OUTPUT_DIR, "dataset_same_character_diff_sources.png")
    plot_same_character_diff_sources(
        df,
        out2,
        metadata_rows,
        character_name_col=character_name_col,
        source_name_col=source_name_col,
        nrows=4,
        ncols=4
    )
    print("已保存:", out2)

    # 3) 同一来源，不同字符
    out3 = os.path.join(OUTPUT_DIR, "dataset_same_source_diff_characters.png")
    plot_same_source_diff_characters(
        df,
        out3,
        metadata_rows,
        character_name_col=character_name_col,
        source_name_col=source_name_col,
        nrows=4,
        ncols=4
    )
    print("已保存:", out3)

    # 4) 元数据
    meta_path = os.path.join(OUTPUT_DIR, "dataset_example_metadata.csv")
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print("已保存:", meta_path)

    print("\n全部完成，输出目录：")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
