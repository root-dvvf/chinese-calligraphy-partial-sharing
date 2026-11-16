import os
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from tqdm import tqdm
from sklearn.metrics import f1_score


# =========================================================
# 1. 配置
# =========================================================

SEEDS = [42, 123, 2026]

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

TEST_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_test.csv"
)

RESULT_DIR = (
    "/root/calligraphy_sci/"
    "results/final_test"
)

BATCH_SIZE = 128
NUM_WORKERS = 8
IMAGE_SIZE = 224

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# =========================================================
# 2. Dataset
# =========================================================

class DualTestDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(
            csv_file
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        idx
    ):

        row = self.data.iloc[idx]

        image_path = str(
            row["path"]
        )

        character_label = int(
            row["character_label"]
        )

        source_label = int(
            row["source_label"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            character_label,
            source_label,
            idx
        )


# =========================================================
# 3. Test预处理
#
# Test绝对不能使用随机数据增强
# =========================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


test_dataset = DualTestDataset(
    TEST_CSV,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


print(
    "Test Samples:",
    len(test_dataset)
)


# =========================================================
# 4. Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "使用设备:",
    device
)


# =========================================================
# 5. Baseline ResNet18
# =========================================================

def build_baseline():

    model = models.resnet18(
        weights=None
    )

    feature_dim = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        feature_dim,
        NUM_CHARACTER_CLASSES
    )

    return model


# =========================================================
# 6. Partial Sharing模型
#
# 最终冻结结构：
#
# conv1～layer3共享
# layer4任务专属
# =========================================================

class PartialSharingLayer3(nn.Module):

    def __init__(
        self,
        num_character_classes,
        num_source_classes
    ):

        super().__init__()


        base = models.resnet18(
            weights=None
        )


        # -------------------------------------------------
        # Shared
        # -------------------------------------------------

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3


        # -------------------------------------------------
        # Task-specific layer4
        # -------------------------------------------------

        import copy

        self.character_layer4 = copy.deepcopy(
            base.layer4
        )

        self.source_layer4 = copy.deepcopy(
            base.layer4
        )


        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        self.character_head = nn.Linear(
            512,
            num_character_classes
        )


        self.source_head = nn.Linear(
            512,
            num_source_classes
        )


    def forward(
        self,
        x
    ):

        # Shared backbone

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        # Character branch

        char_x = self.character_layer4(
            x
        )

        char_x = self.avgpool(
            char_x
        )

        char_x = torch.flatten(
            char_x,
            1
        )


        # Source branch

        source_x = self.source_layer4(
            x
        )

        source_x = self.avgpool(
            source_x
        )

        source_x = torch.flatten(
            source_x,
            1
        )


        character_logits = (
            self.character_head(
                char_x
            )
        )


        source_logits = (
            self.source_head(
                source_x
            )
        )


        return (
            character_logits,
            source_logits
        )


# =========================================================
# 7. Baseline测试函数
# =========================================================

@torch.no_grad()
def test_baseline(
    model,
    seed
):

    model.eval()


    top1_correct = 0
    top5_correct = 0
    total = 0


    all_labels = []
    all_predictions = []


    prediction_rows = []


    for (
        images,
        character_labels,
        source_labels,
        indices

    ) in tqdm(
        test_loader,
        desc=f"Baseline seed={seed}"
    ):


        images = images.to(
            device,
            non_blocking=True
        )


        character_labels = (
            character_labels.to(
                device,
                non_blocking=True
            )
        )


        outputs = model(
            images
        )


        # Top-1

        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        top1_correct += (
            predictions
            .eq(character_labels)
            .sum()
            .item()
        )


        # Top-5

        _, top5 = (
            outputs.topk(
                5,
                dim=1,
                largest=True,
                sorted=True
            )
        )


        top5_correct += (

            top5.eq(

                character_labels
                .view(-1, 1)
                .expand_as(
                    top5
                )

            )
            .sum()
            .item()
        )


        batch_size = (
            images.size(0)
        )

        total += batch_size


        labels_cpu = (
            character_labels
            .cpu()
            .numpy()
        )


        preds_cpu = (
            predictions
            .cpu()
            .numpy()
        )


        indices_cpu = (
            indices
            .cpu()
            .numpy()
        )


        all_labels.extend(
            labels_cpu
        )

        all_predictions.extend(
            preds_cpu
        )


        for i in range(
            batch_size
        ):

            prediction_rows.append({

                "index":
                    int(indices_cpu[i]),

                "character_true":
                    int(labels_cpu[i]),

                "character_pred":
                    int(preds_cpu[i]),

                "correct":
                    int(
                        labels_cpu[i]
                        == preds_cpu[i]
                    )
            })


    top1 = (
        100.0
        * top1_correct
        / total
    )


    top5 = (
        100.0
        * top5_correct
        / total
    )


    macro_f1 = f1_score(

        all_labels,

        all_predictions,

        average="macro",

        zero_division=0
    )


    prediction_path = os.path.join(

        RESULT_DIR,

        f"baseline_seed{seed}_predictions.csv"
    )


    pd.DataFrame(
        prediction_rows
    ).to_csv(

        prediction_path,

        index=False,

        encoding="utf-8-sig"
    )


    return {

        "model":
            "ResNet18 Baseline",

        "seed":
            seed,

        "character_top1":
            top1,

        "character_top5":
            top5,

        "character_macro_f1":
            macro_f1,

        "source_top1":
            np.nan,

        "source_macro_f1":
            np.nan
    }


# =========================================================
# 8. Partial Sharing测试函数
# =========================================================

@torch.no_grad()
def test_partial(
    model,
    seed
):

    model.eval()


    character_top1_correct = 0
    character_top5_correct = 0

    source_correct = 0

    total = 0


    all_character_labels = []
    all_character_predictions = []

    all_source_labels = []
    all_source_predictions = []


    prediction_rows = []


    for (
        images,
        character_labels,
        source_labels,
        indices

    ) in tqdm(
        test_loader,
        desc=f"Partial seed={seed}"
    ):


        images = images.to(
            device,
            non_blocking=True
        )


        character_labels = (
            character_labels.to(
                device,
                non_blocking=True
            )
        )


        source_labels = (
            source_labels.to(
                device,
                non_blocking=True
            )
        )


        (
            character_outputs,
            source_outputs

        ) = model(
            images
        )


        # -------------------------------------------------
        # Character Top-1
        # -------------------------------------------------

        character_predictions = (
            character_outputs.argmax(
                dim=1
            )
        )


        character_top1_correct += (
            character_predictions
            .eq(character_labels)
            .sum()
            .item()
        )


        # -------------------------------------------------
        # Character Top-5
        # -------------------------------------------------

        _, top5 = (
            character_outputs.topk(
                5,
                dim=1,
                largest=True,
                sorted=True
            )
        )


        character_top5_correct += (

            top5.eq(

                character_labels
                .view(-1, 1)
                .expand_as(
                    top5
                )

            )
            .sum()
            .item()
        )


        # -------------------------------------------------
        # Source Top-1
        # -------------------------------------------------

        source_predictions = (
            source_outputs.argmax(
                dim=1
            )
        )


        source_correct += (
            source_predictions
            .eq(source_labels)
            .sum()
            .item()
        )


        batch_size = (
            images.size(0)
        )

        total += batch_size


        char_labels_cpu = (
            character_labels
            .cpu()
            .numpy()
        )


        char_preds_cpu = (
            character_predictions
            .cpu()
            .numpy()
        )


        source_labels_cpu = (
            source_labels
            .cpu()
            .numpy()
        )


        source_preds_cpu = (
            source_predictions
            .cpu()
            .numpy()
        )


        indices_cpu = (
            indices
            .cpu()
            .numpy()
        )


        all_character_labels.extend(
            char_labels_cpu
        )

        all_character_predictions.extend(
            char_preds_cpu
        )


        all_source_labels.extend(
            source_labels_cpu
        )

        all_source_predictions.extend(
            source_preds_cpu
        )


        for i in range(
            batch_size
        ):

            prediction_rows.append({

                "index":
                    int(indices_cpu[i]),

                "character_true":
                    int(char_labels_cpu[i]),

                "character_pred":
                    int(char_preds_cpu[i]),

                "character_correct":
                    int(
                        char_labels_cpu[i]
                        == char_preds_cpu[i]
                    ),

                "source_true":
                    int(source_labels_cpu[i]),

                "source_pred":
                    int(source_preds_cpu[i]),

                "source_correct":
                    int(
                        source_labels_cpu[i]
                        == source_preds_cpu[i]
                    )
            })


    character_top1 = (

        100.0
        * character_top1_correct
        / total
    )


    character_top5 = (

        100.0
        * character_top5_correct
        / total
    )


    character_macro_f1 = f1_score(

        all_character_labels,

        all_character_predictions,

        average="macro",

        zero_division=0
    )


    source_top1 = (

        100.0
        * source_correct
        / total
    )


    source_macro_f1 = f1_score(

        all_source_labels,

        all_source_predictions,

        average="macro",

        zero_division=0
    )


    prediction_path = os.path.join(

        RESULT_DIR,

        f"partial_seed{seed}_predictions.csv"
    )


    pd.DataFrame(
        prediction_rows
    ).to_csv(

        prediction_path,

        index=False,

        encoding="utf-8-sig"
    )


    return {

        "model":
            "Partial Sharing layer3",

        "seed":
            seed,

        "character_top1":
            character_top1,

        "character_top5":
            character_top5,

        "character_macro_f1":
            character_macro_f1,

        "source_top1":
            source_top1,

        "source_macro_f1":
            source_macro_f1
    }


# =========================================================
# 9. 测试三个随机种子的Baseline
# =========================================================

results = []


for seed in SEEDS:

    checkpoint_path = (

        f"/root/calligraphy_sci/"
        f"checkpoints/"
        f"baseline_resnet18_seed{seed}/"
        f"best_model.pth"
    )


    print()
    print(
        "========================================"
    )

    print(
        f"Testing Baseline Seed {seed}"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )


    model = build_baseline().to(
        device
    )


    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )


    model.load_state_dict(
        state_dict
    )


    result = test_baseline(
        model,
        seed
    )


    results.append(
        result
    )


    print(
        f"Character Top-1: "
        f"{result['character_top1']:.4f}%"
    )

    print(
        f"Character Top-5: "
        f"{result['character_top5']:.4f}%"
    )

    print(
        f"Character Macro-F1: "
        f"{result['character_macro_f1']:.6f}"
    )


    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 10. 测试三个随机种子的Partial Sharing
# =========================================================

for seed in SEEDS:

    checkpoint_path = (

        f"/root/calligraphy_sci/"
        f"checkpoints/"
        f"partial_layer3_aux010_seed{seed}/"
        f"best_model.pth"
    )


    print()
    print(
        "========================================"
    )

    print(
        f"Testing Partial Sharing Seed {seed}"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )


    model = PartialSharingLayer3(

        NUM_CHARACTER_CLASSES,

        NUM_SOURCE_CLASSES

    ).to(device)


    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )


    model.load_state_dict(
        state_dict
    )


    result = test_partial(
        model,
        seed
    )


    results.append(
        result
    )


    print(
        f"Character Top-1: "
        f"{result['character_top1']:.4f}%"
    )

    print(
        f"Character Top-5: "
        f"{result['character_top5']:.4f}%"
    )

    print(
        f"Character Macro-F1: "
        f"{result['character_macro_f1']:.6f}"
    )


    print(
        f"Source Top-1: "
        f"{result['source_top1']:.4f}%"
    )

    print(
        f"Source Macro-F1: "
        f"{result['source_macro_f1']:.6f}"
    )


    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 11. 保存每个seed的结果
# =========================================================

result_df = pd.DataFrame(
    results
)


detail_path = os.path.join(
    RESULT_DIR,
    "final_test_results.csv"
)


result_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. 计算Mean ± Std
#
# ddof=1：
# 样本标准差
# =========================================================

summary = {}


for model_name in (
    result_df["model"]
    .unique()
):

    df = result_df[
        result_df["model"]
        == model_name
    ]


    model_summary = {}


    for metric in [

        "character_top1",

        "character_top5",

        "character_macro_f1",

        "source_top1",

        "source_macro_f1"

    ]:


        values = (
            df[metric]
            .dropna()
            .values
        )


        if len(values) > 0:

            model_summary[
                metric
            ] = {

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std":
                    float(
                        np.std(
                            values,
                            ddof=1
                        )
                    ),

                "values":
                    [
                        float(x)
                        for x in values
                    ]
            }


    summary[
        model_name
    ] = model_summary


# =========================================================
# 13. 计算Ours相对Baseline提升
# =========================================================

baseline_mean = (

    summary[
        "ResNet18 Baseline"
    ][
        "character_top1"
    ][
        "mean"
    ]
)


ours_mean = (

    summary[
        "Partial Sharing layer3"
    ][
        "character_top1"
    ][
        "mean"
    ]
)


improvement = (
    ours_mean
    - baseline_mean
)


summary[
    "comparison"
] = {

    "baseline_character_top1_mean":
        baseline_mean,

    "ours_character_top1_mean":
        ours_mean,

    "absolute_improvement_percentage_points":
        improvement
}


summary_path = os.path.join(
    RESULT_DIR,
    "final_test_summary.json"
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
# 14. 最终打印
# =========================================================

print()
print(
    "=============================================="
)

print(
    "========== FINAL TEST SUMMARY ================"
)

print(
    "=============================================="
)


for model_name in [

    "ResNet18 Baseline",

    "Partial Sharing layer3"

]:

    print()

    print(
        model_name
    )


    info = summary[
        model_name
    ]


    print(
        "Character Top-1:"
    )

    print(
        f"{info['character_top1']['mean']:.4f}"
        f" ± "
        f"{info['character_top1']['std']:.4f}%"
    )


    print(
        "Character Top-5:"
    )

    print(
        f"{info['character_top5']['mean']:.4f}"
        f" ± "
        f"{info['character_top5']['std']:.4f}%"
    )


    print(
        "Character Macro-F1:"
    )

    print(
        f"{info['character_macro_f1']['mean']:.6f}"
        f" ± "
        f"{info['character_macro_f1']['std']:.6f}"
    )


    if (
        "source_top1"
        in info
    ):

        print(
            "Source Top-1:"
        )

        print(
            f"{info['source_top1']['mean']:.4f}"
            f" ± "
            f"{info['source_top1']['std']:.4f}%"
        )


        print(
            "Source Macro-F1:"
        )

        print(
            f"{info['source_macro_f1']['mean']:.6f}"
            f" ± "
            f"{info['source_macro_f1']['std']:.6f}"
        )


print()

print(
    "Character Top-1平均提升:"
)

print(
    f"{improvement:+.4f} percentage points"
)


print()

print(
    "详细结果:",
    detail_path
)

print(
    "最终摘要:",
    summary_path
)

print()

print(
    "FINAL TEST COMPLETE"
)
