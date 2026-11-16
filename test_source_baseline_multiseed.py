import os
import json
import pandas as pd
import numpy as np

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

ROOT = "/root/calligraphy_sci"

SEEDS = [
    42,
    123,
    2026
]

NUM_SOURCE_CLASSES = 19

TEST_CSV = os.path.join(
    ROOT,
    "dual_splits",
    "dual_test.csv"
)

RESULT_DIR = os.path.join(
    ROOT,
    "results",
    "source_baseline_final_test"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

BATCH_SIZE = 128
NUM_WORKERS = 8
IMAGE_SIZE = 224


# =========================================================
# 2. Dataset
# =========================================================

class SourceTestDataset(Dataset):

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
            source_label,
            idx
        )


# =========================================================
# 3. Test transform
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


test_dataset = SourceTestDataset(
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
    "Device:",
    device
)


# =========================================================
# 5. Source-only ResNet18
# =========================================================

def build_model():

    model = models.resnet18(
        weights=None
    )

    feature_dim = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        feature_dim,
        NUM_SOURCE_CLASSES
    )

    return model


# =========================================================
# 6. 测试函数
# =========================================================

@torch.no_grad()
def evaluate(
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
        labels,
        indices

    ) in tqdm(

        test_loader,

        desc=f"Source-only seed={seed}"
    ):


        images = images.to(
            device,
            non_blocking=True
        )


        labels = labels.to(
            device,
            non_blocking=True
        )


        outputs = model(
            images
        )


        # =================================================
        # Top-1
        # =================================================

        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        top1_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )


        # =================================================
        # Top-5
        # =================================================

        _, top5 = outputs.topk(

            5,

            dim=1,

            largest=True,

            sorted=True
        )


        top5_correct += (

            top5.eq(

                labels
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
            labels
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
                    int(
                        indices_cpu[i]
                    ),

                "source_true":
                    int(
                        labels_cpu[i]
                    ),

                "source_pred":
                    int(
                        preds_cpu[i]
                    ),

                "correct":
                    int(
                        labels_cpu[i]
                        ==
                        preds_cpu[i]
                    )
            })


    # =====================================================
    # Metrics
    # =====================================================

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


    # =====================================================
    # 保存逐样本预测
    # =====================================================

    prediction_path = os.path.join(

        RESULT_DIR,

        f"source_baseline_seed{seed}_predictions.csv"
    )


    pd.DataFrame(
        prediction_rows
    ).to_csv(

        prediction_path,

        index=False,

        encoding="utf-8-sig"
    )


    return {

        "seed":
            seed,

        "source_top1":
            top1,

        "source_top5":
            top5,

        "source_macro_f1":
            macro_f1
    }


# =========================================================
# 7. 三随机种子正式测试
# =========================================================

results = []


for seed in SEEDS:

    print()

    print(
        "=========================================="
    )

    print(
        f"Testing Source-only Seed {seed}"
    )

    print(
        "=========================================="
    )


    checkpoint_path = os.path.join(

        ROOT,

        "checkpoints",

        f"source_baseline_resnet18_seed{seed}",

        "best_model.pth"
    )


    print(
        "Checkpoint:",
        checkpoint_path
    )


    model = build_model().to(
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


    result = evaluate(
        model,
        seed
    )


    results.append(
        result
    )


    print()

    print(
        f"Source Top-1: "
        f"{result['source_top1']:.6f}%"
    )


    print(
        f"Source Top-5: "
        f"{result['source_top5']:.6f}%"
    )


    print(
        f"Source Macro-F1: "
        f"{result['source_macro_f1']:.8f}"
    )


    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# =========================================================
# 8. 保存逐seed结果
# =========================================================

result_df = pd.DataFrame(
    results
)


result_csv = os.path.join(

    RESULT_DIR,

    "source_baseline_test_results.csv"
)


result_df.to_csv(

    result_csv,

    index=False,

    encoding="utf-8-sig"
)


# =========================================================
# 9. Mean ± Std
#
# ddof=1 = sample standard deviation
# =========================================================

top1_values = (
    result_df[
        "source_top1"
    ].values
)


top5_values = (
    result_df[
        "source_top5"
    ].values
)


f1_values = (
    result_df[
        "source_macro_f1"
    ].values
)


summary = {

    "source_top1": {

        "mean":
            float(
                np.mean(
                    top1_values
                )
            ),

        "std":
            float(
                np.std(
                    top1_values,
                    ddof=1
                )
            ),

        "values":
            [
                float(x)
                for x in top1_values
            ]
    },


    "source_top5": {

        "mean":
            float(
                np.mean(
                    top5_values
                )
            ),

        "std":
            float(
                np.std(
                    top5_values,
                    ddof=1
                )
            ),

        "values":
            [
                float(x)
                for x in top5_values
            ]
    },


    "source_macro_f1": {

        "mean":
            float(
                np.mean(
                    f1_values
                )
            ),

        "std":
            float(
                np.std(
                    f1_values,
                    ddof=1
                )
            ),

        "values":
            [
                float(x)
                for x in f1_values
            ]
    }
}


summary_path = os.path.join(

    RESULT_DIR,

    "source_baseline_test_summary.json"
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
# 10. 最终输出
# =========================================================

print()

print(
    "=============================================="
)

print(
    "SOURCE-ONLY FINAL TEST SUMMARY"
)

print(
    "=============================================="
)


print()

print(
    "Source Top-1:"
)

print(

    f"{summary['source_top1']['mean']:.4f}"
    f" ± "
    f"{summary['source_top1']['std']:.4f}%"
)


print()

print(
    "Source Top-5:"
)

print(

    f"{summary['source_top5']['mean']:.4f}"
    f" ± "
    f"{summary['source_top5']['std']:.4f}%"
)


print()

print(
    "Source Macro-F1:"
)

print(

    f"{summary['source_macro_f1']['mean']:.6f}"
    f" ± "
    f"{summary['source_macro_f1']['std']:.6f}"
)


print()

print(
    "Detailed results:",
    result_csv
)


print(
    "Summary:",
    summary_path
)


print()

print(
    "SOURCE-ONLY FINAL TEST COMPLETE"
)
