import os
import pandas as pd


# =========================================================
# 1. 路径配置
# =========================================================

CALL_ROOT = (
    "/root/calligraphy_sci/data/"
    "calligrapher_dataset_fixed/"
    "chinese-calligraphy-dataset-with-calligrapher"
)

SPLIT_DIR = (
    "/root/calligraphy_sci/splits"
)

OUT_DIR = (
    "/root/calligraphy_sci/dual_splits"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =========================================================
# 2. 扫描第二套书法来源数据集
#
# 建立：
# image_id -> {
#     gif_path,
#     character_name,
#     source_name,
#     style_name
# }
# =========================================================

print(
    "正在扫描 Calligraphic Source 数据集..."
)


id_records = {}

source_names = set()
style_names = set()


for source_folder in sorted(
    os.listdir(CALL_ROOT)
):

    source_path = os.path.join(
        CALL_ROOT,
        source_folder
    )

    if not os.path.isdir(
        source_path
    ):
        continue


    # -----------------------------------------------------
    # 一级目录示例：
    #
    # 楷-多宝塔
    # 楷-虞世南
    # 行-郑板桥
    # 行-王铎
    # 隶-金农
    #
    # 前半部分 = 书体
    # 完整目录名 = 19类书法来源
    # -----------------------------------------------------

    if "-" in source_folder:

        style_name = (
            source_folder
            .split("-", 1)[0]
        )

    else:

        style_name = "UNKNOWN"


    source_name = (
        source_folder
    )


    source_names.add(
        source_name
    )

    style_names.add(
        style_name
    )


    # -----------------------------------------------------
    # 第二层目录 = 汉字类别
    # -----------------------------------------------------

    for character_name in sorted(
        os.listdir(source_path)
    ):

        character_path = os.path.join(
            source_path,
            character_name
        )

        if not os.path.isdir(
            character_path
        ):
            continue


        # -------------------------------------------------
        # 第三层 = GIF图片
        # -------------------------------------------------

        for filename in os.listdir(
            character_path
        ):

            if not filename.lower().endswith(
                ".gif"
            ):
                continue


            image_id = os.path.splitext(
                filename
            )[0]


            gif_path = os.path.join(
                character_path,
                filename
            )


            # 防止同一个图片ID出现两次
            if image_id in id_records:

                raise RuntimeError(
                    "发现重复图片ID: "
                    + image_id
                )


            id_records[
                image_id
            ] = {

                "image_id":
                    image_id,

                "gif_path":
                    gif_path,

                "character_name":
                    character_name,

                "source_name":
                    source_name,

                "style_name":
                    style_name
            }


print()

print(
    "第二套数据唯一ID数:",
    len(id_records)
)


# =========================================================
# 3. 创建19类来源标签
# =========================================================

source_names = sorted(
    source_names
)

source_to_idx = {

    source_name: idx

    for idx, source_name
    in enumerate(source_names)
}


print()

print(
    "========== 书法来源标签 =========="
)

print(
    "来源类别数:",
    len(source_to_idx)
)

for source_name, label in (
    source_to_idx.items()
):

    print(
        label,
        source_name
    )


# =========================================================
# 4. 创建3类书体标签
# =========================================================

style_names = sorted(
    style_names
)

style_to_idx = {

    style_name: idx

    for idx, style_name
    in enumerate(style_names)
}


print()

print(
    "========== 书体标签 =========="
)

print(
    "书体类别数:",
    len(style_to_idx)
)

for style_name, label in (
    style_to_idx.items()
):

    print(
        label,
        style_name
    )


# =========================================================
# 5. 转换原有 Train / Val / Test
#
# 特别注意：
#
# path 使用第一套数据的 JPG
#
# source_gif_path 只保留第二套 GIF 路径用于追踪
#
# 这样双任务实验和原 ResNet18 Baseline
# 使用完全相同的输入图片。
# =========================================================

def convert_split(
    split_name
):

    input_csv = os.path.join(
        SPLIT_DIR,
        f"{split_name}.csv"
    )


    print()

    print(
        "正在处理:",
        input_csv
    )


    df = pd.read_csv(
        input_csv
    )


    rows = []

    missing_id = 0
    character_mismatch = 0


    for _, row in df.iterrows():

        # -------------------------------------------------
        # 第一套数据原始 JPG 路径
        # -------------------------------------------------

        jpg_path = str(
            row["path"]
        )


        character_label = int(
            row["label"]
        )


        character_name = str(
            row["class_name"]
        )


        # -------------------------------------------------
        # 从文件名提取图片ID
        #
        # 例如：
        # /xxx/砚/107195.jpg
        #
        # image_id = 107195
        # -------------------------------------------------

        filename = os.path.basename(
            jpg_path
        )


        image_id = os.path.splitext(
            filename
        )[0]


        # -------------------------------------------------
        # 第二套数据中是否存在对应图片
        # -------------------------------------------------

        if image_id not in id_records:

            missing_id += 1
            continue


        record = id_records[
            image_id
        ]


        # -------------------------------------------------
        # 再次确认两套数据汉字标签一致
        # -------------------------------------------------

        second_character_name = (
            record[
                "character_name"
            ]
        )


        if (
            second_character_name
            != character_name
        ):

            character_mismatch += 1

            print(
                "汉字标签不一致:",
                image_id,
                character_name,
                second_character_name
            )

            continue


        source_name = (
            record[
                "source_name"
            ]
        )


        style_name = (
            record[
                "style_name"
            ]
        )


        # -------------------------------------------------
        # 写入双任务数据
        # -------------------------------------------------

        rows.append({

            # 图片唯一ID
            "image_id":
                image_id,


            # 正式训练输入
            # 使用第一套数据的JPG
            "path":
                jpg_path,


            # 第二套数据GIF
            # 只作为追踪信息保留
            "source_gif_path":
                record[
                    "gif_path"
                ],


            # 汉字标签
            "character_label":
                character_label,


            # 汉字名称
            "character_name":
                character_name,


            # 19类书法来源标签
            "source_label":
                source_to_idx[
                    source_name
                ],


            # 书法来源名称
            "source_name":
                source_name,


            # 3类书体标签
            "style_label":
                style_to_idx[
                    style_name
                ],


            # 楷 / 行 / 隶
            "style_name":
                style_name
        })


    out_df = pd.DataFrame(
        rows
    )


    output_csv = os.path.join(
        OUT_DIR,
        f"dual_{split_name}.csv"
    )


    out_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )


    print()

    print(
        f"========== {split_name.upper()} =========="
    )

    print(
        "原始样本数:",
        len(df)
    )

    print(
        "双任务样本数:",
        len(out_df)
    )

    print(
        "缺失ID:",
        missing_id
    )

    print(
        "汉字标签不一致:",
        character_mismatch
    )

    print(
        "输出:",
        output_csv
    )


    return out_df


# =========================================================
# 6. 构建 Train / Val / Test
# =========================================================

train_df = convert_split(
    "train"
)

val_df = convert_split(
    "val"
)

test_df = convert_split(
    "test"
)


# =========================================================
# 7. 保存19类来源映射
# =========================================================

source_mapping_df = pd.DataFrame({

    "source_label":
        list(
            source_to_idx.values()
        ),

    "source_name":
        list(
            source_to_idx.keys()
        )
})


source_mapping_path = os.path.join(
    OUT_DIR,
    "source_mapping.csv"
)


source_mapping_df.to_csv(
    source_mapping_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. 保存3类书体映射
# =========================================================

style_mapping_df = pd.DataFrame({

    "style_label":
        list(
            style_to_idx.values()
        ),

    "style_name":
        list(
            style_to_idx.keys()
        )
})


style_mapping_path = os.path.join(
    OUT_DIR,
    "style_mapping.csv"
)


style_mapping_df.to_csv(
    style_mapping_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. 保存来源训练集分布
# =========================================================

source_distribution = (

    train_df[
        "source_name"
    ]

    .value_counts()

    .rename_axis(
        "source_name"
    )

    .reset_index(
        name="train_samples"
    )
)


source_distribution_path = os.path.join(
    OUT_DIR,
    "source_distribution.csv"
)


source_distribution.to_csv(
    source_distribution_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. 保存书体训练集分布
# =========================================================

style_distribution = (

    train_df[
        "style_name"
    ]

    .value_counts()

    .rename_axis(
        "style_name"
    )

    .reset_index(
        name="train_samples"
    )
)


style_distribution_path = os.path.join(
    OUT_DIR,
    "style_distribution.csv"
)


style_distribution.to_csv(
    style_distribution_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. 最终统计
# =========================================================

total_samples = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)


print()

print(
    "========== 双任务数据集完成 =========="
)


print(
    "Train:",
    len(train_df)
)

print(
    "Val:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

print(
    "Total:",
    total_samples
)


print()

print(
    "汉字类别数:",
    train_df[
        "character_label"
    ].nunique()
)

print(
    "来源类别数:",
    train_df[
        "source_label"
    ].nunique()
)

print(
    "书体类别数:",
    train_df[
        "style_label"
    ].nunique()
)


print()

print(
    "========== 训练集来源分布 =========="
)

print(
    source_distribution.to_string(
        index=False
    )
)


print()

print(
    "========== 训练集书体分布 =========="
)

print(
    style_distribution.to_string(
        index=False
    )
)


print()

print(
    "========== 输出文件 =========="
)

print(
    os.path.join(
        OUT_DIR,
        "dual_train.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "dual_val.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "dual_test.csv"
    )
)

print(
    source_mapping_path
)

print(
    style_mapping_path
)

print(
    source_distribution_path
)

print(
    style_distribution_path
)
