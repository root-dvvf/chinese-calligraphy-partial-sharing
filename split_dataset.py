import os
import csv
import random

DATA_DIR = "/root/calligraphy_sci/data/chinese-calligraphy-dataset/chinese-calligraphy-dataset"
OUT_DIR = "/root/calligraphy_sci/splits"

MIN_SAMPLES = 5
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------
# 1. 找到样本数 >= 5 的类别
# ------------------------------------------------

valid_classes = []

for cls_name in os.listdir(DATA_DIR):
    cls_dir = os.path.join(DATA_DIR, cls_name)

    if not os.path.isdir(cls_dir):
        continue

    images = [
        f for f in os.listdir(cls_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) >= MIN_SAMPLES:
        valid_classes.append(cls_name)

valid_classes = sorted(valid_classes)

# 固定 label 映射
class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(valid_classes)
}

print("保留类别数:", len(valid_classes))

train_rows = []
val_rows = []
test_rows = []

# ------------------------------------------------
# 2. 每个类别单独划分
# ------------------------------------------------

for cls_name in valid_classes:

    cls_dir = os.path.join(DATA_DIR, cls_name)

    images = [
        f for f in os.listdir(cls_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # 为每个类别使用稳定随机顺序
    random.shuffle(images)

    n = len(images)

    # 验证集约 10%
    n_val = max(1, round(n * 0.10))

    # 测试集约 10%
    n_test = max(1, round(n * 0.10))

    # 剩下全部用于训练
    n_train = n - n_val - n_test

    # n >= 5，所以训练集至少有 3 张
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    label = class_to_idx[cls_name]

    for img in train_imgs:
        train_rows.append([
            os.path.join(cls_dir, img),
            label,
            cls_name
        ])

    for img in val_imgs:
        val_rows.append([
            os.path.join(cls_dir, img),
            label,
            cls_name
        ])

    for img in test_imgs:
        test_rows.append([
            os.path.join(cls_dir, img),
            label,
            cls_name
        ])


# ------------------------------------------------
# 3. 保存 CSV
# ------------------------------------------------

def save_csv(filename, rows):

    path = os.path.join(OUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow([
            "path",
            "label",
            "class_name"
        ])

        writer.writerows(rows)

    return path


train_path = save_csv("train.csv", train_rows)
val_path = save_csv("val.csv", val_rows)
test_path = save_csv("test.csv", test_rows)


# ------------------------------------------------
# 4. 保存类别映射
# ------------------------------------------------

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

    for cls_name, label in class_to_idx.items():
        writer.writerow([
            label,
            cls_name
        ])


# ------------------------------------------------
# 5. 输出统计
# ------------------------------------------------

total = (
    len(train_rows)
    + len(val_rows)
    + len(test_rows)
)

print()
print("Train:", len(train_rows))
print("Val:", len(val_rows))
print("Test:", len(test_rows))
print("Total:", total)

print()
print("Train比例:", len(train_rows) / total)
print("Val比例:", len(val_rows) / total)
print("Test比例:", len(test_rows) / total)

print()
print("输出文件:")
print(train_path)
print(val_path)
print(test_path)
print(mapping_path)
