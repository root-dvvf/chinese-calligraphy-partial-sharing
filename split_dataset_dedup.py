import os
import csv
import random
import hashlib
from collections import defaultdict


DATA_DIR = (
    "/root/calligraphy_sci/data/"
    "chinese-calligraphy-dataset/"
    "chinese-calligraphy-dataset"
)

OUT_DIR = "/root/calligraphy_sci/splits"

MIN_SAMPLES = 5
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# 1. MD5
# =========================================================

def file_md5(path):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# =========================================================
# 2. 扫描整个数据集
# =========================================================

print("正在扫描全部图片...")

hash_records = defaultdict(list)

total_images = 0

for cls_name in sorted(os.listdir(DATA_DIR)):

    cls_dir = os.path.join(
        DATA_DIR,
        cls_name
    )

    if not os.path.isdir(cls_dir):
        continue

    for filename in os.listdir(cls_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(
            cls_dir,
            filename
        )

        h = file_md5(path)

        hash_records[h].append(
            (path, cls_name)
        )

        total_images += 1


print("原始图片数:", total_images)
print("唯一MD5数量:", len(hash_records))


# =========================================================
# 3. 全局去重
# =========================================================

class_images = defaultdict(list)

same_class_duplicates = 0
cross_class_conflicts = 0
conflict_images = 0


for h, records in hash_records.items():

    classes = set(
        cls_name
        for _, cls_name in records
    )

    # ---------------------------------------------
    # 情况1：
    # 相同图片只属于同一个类别
    # 保留一张
    # ---------------------------------------------

    if len(classes) == 1:

        path, cls_name = records[0]

        class_images[cls_name].append(path)

        if len(records) > 1:

            same_class_duplicates += (
                len(records) - 1
            )

    # ---------------------------------------------
    # 情况2：
    # 相同图片属于不同类别
    # 标签冲突，全部删除
    # ---------------------------------------------

    else:

        cross_class_conflicts += 1
        conflict_images += len(records)


print()
print("========== 去重统计 ==========")

print(
    "同类别重复、被删除图片数:",
    same_class_duplicates
)

print(
    "跨类别冲突MD5组数:",
    cross_class_conflicts
)

print(
    "跨类别冲突图片总数:",
    conflict_images
)


# =========================================================
# 4. 删除去重后不足5张的类别
# =========================================================

valid_class_images = {}

removed_small_classes = 0

for cls_name, images in class_images.items():

    if len(images) >= MIN_SAMPLES:

        valid_class_images[
            cls_name
        ] = images

    else:

        removed_small_classes += 1


valid_classes = sorted(
    valid_class_images.keys()
)


print()
print(
    "去重后保留类别数:",
    len(valid_classes)
)

print(
    "去重后不足5张、被移除类别数:",
    removed_small_classes
)


# =========================================================
# 5. 创建标签映射
# =========================================================

class_to_idx = {

    cls_name: idx

    for idx, cls_name
    in enumerate(valid_classes)
}


# =========================================================
# 6. 分层划分
# =========================================================

train_rows = []
val_rows = []
test_rows = []


for cls_name in valid_classes:

    images = (
        valid_class_images[
            cls_name
        ].copy()
    )

    random.shuffle(images)

    n = len(images)

    n_val = max(
        1,
        round(n * 0.10)
    )

    n_test = max(
        1,
        round(n * 0.10)
    )

    n_train = (
        n
        - n_val
        - n_test
    )

    train_imgs = (
        images[:n_train]
    )

    val_imgs = (
        images[
            n_train:
            n_train + n_val
        ]
    )

    test_imgs = (
        images[
            n_train + n_val:
        ]
    )

    label = class_to_idx[
        cls_name
    ]

    for path in train_imgs:

        train_rows.append([
            path,
            label,
            cls_name
        ])

    for path in val_imgs:

        val_rows.append([
            path,
            label,
            cls_name
        ])

    for path in test_imgs:

        test_rows.append([
            path,
            label,
            cls_name
        ])


# =========================================================
# 7. 保存CSV
# =========================================================

def save_csv(filename, rows):

    path = os.path.join(
        OUT_DIR,
        filename
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "path",
            "label",
            "class_name"
        ])

        writer.writerows(rows)

    return path


train_path = save_csv(
    "train.csv",
    train_rows
)

val_path = save_csv(
    "val.csv",
    val_rows
)

test_path = save_csv(
    "test.csv",
    test_rows
)


# =========================================================
# 8. 保存标签映射
# =========================================================

mapping_path = os.path.join(
    OUT_DIR,
    "class_mapping.csv"
)

with open(
    mapping_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "label",
        "class_name"
    ])

    for cls_name, label in (
        class_to_idx.items()
    ):

        writer.writerow([
            label,
            cls_name
        ])


# =========================================================
# 9. 输出最终划分
# =========================================================

total = (
    len(train_rows)
    + len(val_rows)
    + len(test_rows)
)

print()
print(
    "========== Final Split =========="
)

print(
    "类别数:",
    len(valid_classes)
)

print(
    "Train:",
    len(train_rows)
)

print(
    "Val:",
    len(val_rows)
)

print(
    "Test:",
    len(test_rows)
)

print(
    "Total:",
    total
)

print()

print(
    "Train比例:",
    len(train_rows) / total
)

print(
    "Val比例:",
    len(val_rows) / total
)

print(
    "Test比例:",
    len(test_rows) / total
)
