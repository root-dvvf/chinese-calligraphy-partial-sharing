import os
import json
import csv
import time
import random
import argparse

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
# 1. 命令行参数
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Random seed"
)

args = parser.parse_args()

SEED = args.seed


# =========================================================
# 2. 基础配置
# =========================================================

NUM_SOURCE_CLASSES = 19

TRAIN_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_train.csv"
)

VAL_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_val.csv"
)

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224


EXPERIMENT_NAME = (
    f"source_baseline_resnet18_seed{SEED}"
)

RESULT_DIR = os.path.join(
    "/root/calligraphy_sci/results",
    EXPERIMENT_NAME
)

CHECKPOINT_DIR = os.path.join(
    "/root/calligraphy_sci/checkpoints",
    EXPERIMENT_NAME
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


print()

print(
    "========================================"
)

print(
    "实验:",
    EXPERIMENT_NAME
)

print(
    "模型: ResNet18 Source-only Baseline"
)

print(
    "任务: Source Recognition"
)

print(
    "Source Classes:",
    NUM_SOURCE_CLASSES
)

print(
    "Seed:",
    SEED
)

print(
    "========================================"
)


# =========================================================
# 3. 固定随机种子
# =========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)


# =========================================================
# 4. Dataset
#
# 和最终多任务模型使用完全相同的 dual_splits
# 这里只读取 source_label
# =========================================================

class SourceDataset(Dataset):

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
            source_label
        )


# =========================================================
# 5. 数据增强
#
# 与 Character Baseline / Partial Sharing 一致
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(5),

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


val_transform = transforms.Compose([

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


# =========================================================
# 6. Dataset / DataLoader
# =========================================================

train_dataset = SourceDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = SourceDataset(
    VAL_CSV,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


print()

print(
    "Train Samples:",
    len(train_dataset)
)

print(
    "Val Samples:",
    len(val_dataset)
)


# =========================================================
# 7. ResNet18 Source-only
# =========================================================

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


# =========================================================
# 8. GPU
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

model = model.to(
    device
)


# =========================================================
# 9. 参数量
# =========================================================

total_params = sum(

    p.numel()

    for p in model.parameters()
)


trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)


print(
    "模型参数量:",
    total_params
)

print(
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 10. Loss
#
# Source-only baseline保持最标准的CrossEntropy
# 不引入类别权重
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 11. Optimizer
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# =========================================================
# 12. Scheduler
# =========================================================

scheduler = (

    torch.optim.lr_scheduler
    .CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 13. 保存配置
# =========================================================

config = {

    "experiment":
        EXPERIMENT_NAME,

    "architecture":
        "ResNet18",

    "task":
        "Source Recognition",

    "num_source_classes":
        NUM_SOURCE_CLASSES,

    "pretrained":
        False,

    "train_csv":
        TRAIN_CSV,

    "val_csv":
        VAL_CSV,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        NUM_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "optimizer":
        "AdamW",

    "scheduler":
        "CosineAnnealingLR",

    "image_size":
        IMAGE_SIZE,

    "augmentation":
        "RandomRotation(5)",

    "loss":
        "CrossEntropyLoss",

    "model_selection":
        "Val Source Top-1",

    "seed":
        SEED,

    "total_params":
        total_params,

    "trainable_params":
        trainable_params
}


with open(

    os.path.join(
        RESULT_DIR,
        "config.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        config,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# 14. 单轮训练
# =========================================================

def train_one_epoch(
    model,
    loader
):

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    loop = tqdm(
        loader,
        desc="Train"
    )


    for (
        images,
        labels

    ) in loop:


        images = images.to(
            device,
            non_blocking=True
        )


        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        batch_size = (
            images.size(0)
        )


        running_loss += (
            loss.item()
            * batch_size
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )


        total += batch_size


        loop.set_postfix(

            acc=(
                f"{100.0 * correct / total:.2f}"
            ),

            loss=(
                f"{loss.item():.4f}"
            )
        )


    train_loss = (
        running_loss
        / total
    )


    train_acc = (
        100.0
        * correct
        / total
    )


    return (
        train_loss,
        train_acc
    )


# =========================================================
# 15. 验证
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()


    running_loss = 0.0

    top1_correct = 0

    top5_correct = 0

    total = 0


    all_labels = []

    all_predictions = []


    loop = tqdm(
        loader,
        desc="Val"
    )


    for (
        images,
        labels

    ) in loop:


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


        loss = criterion(
            outputs,
            labels
        )


        batch_size = (
            images.size(0)
        )


        running_loss += (
            loss.item()
            * batch_size
        )


        # -------------------------------------------------
        # Top-1
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Top-5
        # -------------------------------------------------

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


        total += batch_size


        all_labels.extend(

            labels
            .cpu()
            .numpy()
        )


        all_predictions.extend(

            predictions
            .cpu()
            .numpy()
        )


    val_loss = (
        running_loss
        / total
    )


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


    return (
        val_loss,
        top1,
        top5,
        macro_f1
    )


# =========================================================
# 16. 正式训练
# =========================================================

best_top1 = -1.0

best_epoch = 0

best_top5 = 0.0

best_macro_f1 = 0.0

best_val_loss = 0.0


log_rows = []


log_path = os.path.join(

    RESULT_DIR,

    "train_log.csv"
)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print()

    print(
        f"========== Epoch "
        f"{epoch}/{NUM_EPOCHS} "
        f"=========="
    )


    start_time = time.time()


    (
        train_loss,
        train_acc

    ) = train_one_epoch(

        model,

        train_loader
    )


    (
        val_loss,
        val_top1,
        val_top5,
        val_macro_f1

    ) = validate(

        model,

        val_loader
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    scheduler.step()


    elapsed = (
        time.time()
        - start_time
    )


    print()

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Source Acc: "
        f"{train_acc:.2f}%"
    )


    print()

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Source Top-1: "
        f"{val_top1:.4f}%"
    )

    print(
        f"Val Source Top-5: "
        f"{val_top5:.4f}%"
    )

    print(
        f"Val Source Macro-F1: "
        f"{val_macro_f1:.6f}"
    )


    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{elapsed:.1f}s"
    )


    # =====================================================
    # 保存日志
    # =====================================================

    log_rows.append([

        epoch,

        train_loss,

        train_acc,

        val_loss,

        val_top1,

        val_top5,

        val_macro_f1,

        current_lr,

        elapsed
    ])


    with open(

        log_path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:


        writer = csv.writer(f)


        writer.writerow([

            "epoch",

            "train_loss",

            "train_source_acc",

            "val_loss",

            "val_source_top1",

            "val_source_top5",

            "val_source_macro_f1",

            "learning_rate",

            "epoch_time"
        ])


        writer.writerows(
            log_rows
        )


    # =====================================================
    # 保存最佳模型
    #
    # Source-only模型严格按照Val Source Top-1选择
    # =====================================================

    if val_top1 > best_top1:

        best_top1 = (
            val_top1
        )

        best_epoch = (
            epoch
        )

        best_top5 = (
            val_top5
        )

        best_macro_f1 = (
            val_macro_f1
        )

        best_val_loss = (
            val_loss
        )


        torch.save(

            model.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                "best_model.pth"
            )
        )


        print()

        print(
            "保存新的最佳Source-only模型"
        )


# =========================================================
# 17. 保存最佳结果
# =========================================================

summary = {

    "experiment":
        EXPERIMENT_NAME,

    "architecture":
        "ResNet18",

    "task":
        "Source Recognition",

    "seed":
        SEED,

    "best_epoch":
        best_epoch,

    "val_source_top1":
        best_top1,

    "val_source_top5":
        best_top5,

    "val_source_macro_f1":
        best_macro_f1,

    "val_loss":
        best_val_loss,

    "total_params":
        total_params
}


summary_path = os.path.join(

    RESULT_DIR,

    "best_summary.json"
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
# 18. 最终输出
# =========================================================

print()

print(
    "=========================================="
)

print(
    "Source-only Baseline训练完成"
)

print(
    "=========================================="
)


print(
    "Seed:",
    SEED
)


print(
    "最佳 Epoch:",
    best_epoch
)


print()

print(
    "Val Source Top-1:",
    best_top1
)

print(
    "Val Source Top-5:",
    best_top5
)

print(
    "Val Source Macro-F1:",
    best_macro_f1
)


print()

print(
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


print()

print(
    "训练日志:",
    log_path
)


print(
    "最佳模型:",
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )
)


print(
    "最佳结果摘要:",
    summary_path
)
