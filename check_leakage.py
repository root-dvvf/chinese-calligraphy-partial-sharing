import hashlib
import pandas as pd
from tqdm import tqdm


TRAIN_CSV = "/root/calligraphy_sci/splits/train.csv"
VAL_CSV = "/root/calligraphy_sci/splits/val.csv"
TEST_CSV = "/root/calligraphy_sci/splits/test.csv"


def file_md5(path):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


def load_hashes(csv_path, name):

    df = pd.read_csv(csv_path)

    hashes = {}

    print(f"\n正在检查 {name} ...")

    for path in tqdm(df["path"]):

        h = file_md5(path)

        hashes.setdefault(h, []).append(path)

    return hashes


train_hashes = load_hashes(
    TRAIN_CSV,
    "Train"
)

val_hashes = load_hashes(
    VAL_CSV,
    "Val"
)

test_hashes = load_hashes(
    TEST_CSV,
    "Test"
)


train_set = set(train_hashes.keys())
val_set = set(val_hashes.keys())
test_set = set(test_hashes.keys())


train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set


print()
print("========== Leakage Check ==========")

print(
    "Train-Val 完全重复图片:",
    len(train_val_overlap)
)

print(
    "Train-Test 完全重复图片:",
    len(train_test_overlap)
)

print(
    "Val-Test 完全重复图片:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print()
    print("未发现完全重复图片，MD5检查通过。")

else:

    print()
    print("发现重复图片，需要进一步处理。")
