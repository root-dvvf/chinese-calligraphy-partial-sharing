import os
from collections import defaultdict, Counter


CHAR_ROOT = (
    "/root/calligraphy_sci/data/"
    "chinese-calligraphy-dataset/"
    "chinese-calligraphy-dataset"
)

CALL_ROOT = (
    "/root/calligraphy_sci/data/"
    "calligrapher_dataset_fixed/"
    "chinese-calligraphy-dataset-with-calligrapher"
)


# =========================================================
# 1. 扫描第一个 Character 数据集
# =========================================================

print("正在扫描 Character 数据集...")

char_records = defaultdict(list)

for char_name in os.listdir(CHAR_ROOT):

    char_dir = os.path.join(
        CHAR_ROOT,
        char_name
    )

    if not os.path.isdir(char_dir):
        continue

    for filename in os.listdir(char_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_id = os.path.splitext(
            filename
        )[0]

        char_records[image_id].append({
            "character": char_name,
            "path": os.path.join(
                char_dir,
                filename
            )
        })


# =========================================================
# 2. 扫描 Calligrapher 数据集
# =========================================================

print("正在扫描 Calligrapher 数据集...")

call_records = defaultdict(list)

style_counter = Counter()
calligrapher_counter = Counter()
folder_counter = Counter()


for folder_name in os.listdir(CALL_ROOT):

    folder_path = os.path.join(
        CALL_ROOT,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue

    # 例如：
    # 行-郑板桥
    # 楷-虞世南

    if "-" in folder_name:

        style, calligrapher = (
            folder_name.split("-", 1)
        )

    else:

        style = "UNKNOWN"
        calligrapher = folder_name

    folder_counter[folder_name] += 1
    style_counter[style] += 1
    calligrapher_counter[calligrapher] += 1


    for char_name in os.listdir(
        folder_path
    ):

        char_dir = os.path.join(
            folder_path,
            char_name
        )

        if not os.path.isdir(char_dir):
            continue


        for filename in os.listdir(
            char_dir
        ):

            if not filename.lower().endswith(
                ".gif"
            ):
                continue

            image_id = os.path.splitext(
                filename
            )[0]

            call_records[image_id].append({

                "style":
                    style,

                "calligrapher":
                    calligrapher,

                "character":
                    char_name,

                "path":
                    os.path.join(
                        char_dir,
                        filename
                    )
            })


# =========================================================
# 3. ID统计
# =========================================================

char_ids = set(
    char_records.keys()
)

call_ids = set(
    call_records.keys()
)

common_ids = (
    char_ids
    & call_ids
)

only_char = (
    char_ids
    - call_ids
)

only_call = (
    call_ids
    - char_ids
)


print()
print(
    "========== ID匹配统计 =========="
)

print(
    "Character唯一ID数:",
    len(char_ids)
)

print(
    "Calligrapher唯一ID数:",
    len(call_ids)
)

print(
    "共同ID数:",
    len(common_ids)
)

print(
    "仅Character存在:",
    len(only_char)
)

print(
    "仅Calligrapher存在:",
    len(only_call)
)


if char_ids:

    print(
        "Character匹配率:",
        f"{len(common_ids) / len(char_ids) * 100:.4f}%"
    )

if call_ids:

    print(
        "Calligrapher匹配率:",
        f"{len(common_ids) / len(call_ids) * 100:.4f}%"
    )


# =========================================================
# 4. 检查同一个ID是否出现多次
# =========================================================

char_duplicate_ids = {

    image_id: records

    for image_id, records
    in char_records.items()

    if len(records) > 1
}


call_duplicate_ids = {

    image_id: records

    for image_id, records
    in call_records.items()

    if len(records) > 1
}


print()
print(
    "========== 重复ID检查 =========="
)

print(
    "Character重复ID数:",
    len(char_duplicate_ids)
)

print(
    "Calligrapher重复ID数:",
    len(call_duplicate_ids)
)


# =========================================================
# 5. 检查汉字标签是否一致
# =========================================================

consistent = 0
inconsistent = 0

examples = []


for image_id in common_ids:

    char_labels = set(

        x["character"]

        for x in char_records[
            image_id
        ]
    )

    call_labels = set(

        x["character"]

        for x in call_records[
            image_id
        ]
    )


    if char_labels == call_labels:

        consistent += 1

    else:

        inconsistent += 1

        if len(examples) < 20:

            examples.append({
                "id":
                    image_id,

                "character_dataset":
                    list(char_labels),

                "calligrapher_dataset":
                    list(call_labels)
            })


print()
print(
    "========== 汉字标签一致性 =========="
)

print(
    "一致ID数:",
    consistent
)

print(
    "不一致ID数:",
    inconsistent
)


if common_ids:

    print(
        "标签一致率:",
        f"{consistent / len(common_ids) * 100:.4f}%"
    )


if examples:

    print()
    print(
        "部分不一致样本:"
    )

    for x in examples:

        print(x)


# =========================================================
# 6. 统计书体和书法家
# =========================================================

print()
print(
    "========== 书体统计 =========="
)

print(
    "书体数量:",
    len(style_counter)
)

for name in sorted(
    style_counter.keys()
):

    print(
        name,
        style_counter[name]
    )


print()
print(
    "========== 书法家统计 =========="
)

print(
    "书法家数量:",
    len(calligrapher_counter)
)

for name in sorted(
    calligrapher_counter.keys()
):

    print(name)


print()
print(
    "========== 总结 =========="
)

print(
    "可用于双任务的共同ID:",
    len(common_ids)
)
