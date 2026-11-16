import os
import json
import random
import csv
import time
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
    "--source-weight",
    type=float,
    required=True,
    help="Source auxiliary task loss weight"
)

args = parser.parse_args()

SOURCE_WEIGHT = args.source_weight

if SOURCE_WEIGHT < 0:
    raise ValueError("source-weight 不能小于0")


# =========================================================
# 2. 基本配置
# =========================================================

SEED = 42

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

TRAIN_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_train.csv"
)

VAL_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_val.csv"
)

# 例如：
# 0.10 -> aux010
# 0.25 -> aux025
# 0.50 -> aux050

weight_tag = int(
    round(SOURCE_WEIGHT * 100)
)

EXPERIMENT_NAME = (
    f"multitask_aux{weight_tag:03d}"
)

RESULT_DIR = os.path.join(
    "/root/calligraphy_sci/results",
    EXPERIMENT_NAME
)

CHECKPOINT_DIR = os.path.join(
    "/root/calligraphy_sci/checkpoints",
    EXPERIMENT_NAME
)

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


print(
    "========================================"
)

print(
    "实验:",
    EXPERIMENT_NAME
)

print(
    "Character Weight:",
    1.0
)

print(
    "Source Weight:",
    SOURCE_WEIGHT
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
torch.cuda.manual_seed_all(SEED)


# =========================================================
# 4. Dataset
# =========================================================

class DualTaskDataset(Dataset):

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

        img_path = str(
            row["path"]
        )

        character_label = int(
            row["character_label"]
        )

        source_label = int(
            row["source_label"]
        )

        image = Image.open(
            img_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            character_label,
            source_label
        )


# =========================================================
# 5. 图像预处理
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
# 6. DataLoader
# =========================================================

train_dataset = DualTaskDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = DualTaskDataset(
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


# =========================================================
# 7. 双任务 ResNet18
# =========================================================

class MultiTaskResNet18(nn.Module):

    def __init__(
        self,
        num_character_classes,
        num_source_classes
    ):

        super().__init__()


        self.backbone = models.resnet18(
            weights=None
        )


        feature_dim = (
            self.backbone.fc.in_features
        )


        self.backbone.fc = nn.Identity()


        # 主任务：汉字分类
        self.character_head = nn.Linear(
            feature_dim,
            num_character_classes
        )


        # 辅助任务：书法来源分类
        self.source_head = nn.Linear(
            feature_dim,
            num_source_classes
        )


    def forward(
        self,
        x
    ):

        features = self.backbone(
            x
        )


        character_logits = (
            self.character_head(
                features
            )
        )


        source_logits = (
            self.source_head(
                features
            )
        )


        return (
            character_logits,
            source_logits
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


model = MultiTaskResNet18(
    NUM_CHARACTER_CLASSES,
    NUM_SOURCE_CLASSES
).to(device)


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
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 10. Loss
# =========================================================

character_criterion = nn.CrossEntropyLoss()
source_criterion = nn.CrossEntropyLoss()


# =========================================================
# 11. 优化器与学习率
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 12. 保存配置
# =========================================================

config = {

    "experiment":
        EXPERIMENT_NAME,

    "backbone":
        "ResNet18",

    "pretrained":
        False,

    "primary_task":
        "Character Recognition",

    "primary_classes":
        NUM_CHARACTER_CLASSES,

    "auxiliary_task":
        "Calligraphic Source Classification",

    "auxiliary_classes":
        NUM_SOURCE_CLASSES,

    "character_loss_weight":
        1.0,

    "source_loss_weight":
        SOURCE_WEIGHT,

    "model_selection":
        "Val Character Top-1",

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
# 13. 单轮训练
# =========================================================

def train_one_epoch(
    model,
    loader
):

    model.train()


    running_total_loss = 0.0
    running_character_loss = 0.0
    running_source_loss = 0.0


    character_correct = 0
    source_correct = 0

    total = 0


    loop = tqdm(
        loader,
        desc="Train"
    )


    for (
        images,
        character_labels,
        source_labels

    ) in loop:


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


        optimizer.zero_grad()


        (
            character_outputs,
            source_outputs

        ) = model(
            images
        )


        character_loss = (
            character_criterion(
                character_outputs,
                character_labels
            )
        )


        source_loss = (
            source_criterion(
                source_outputs,
                source_labels
            )
        )


        total_loss = (

            character_loss

            +

            SOURCE_WEIGHT
            * source_loss
        )


        total_loss.backward()

        optimizer.step()


        batch_size = images.size(0)


        running_total_loss += (
            total_loss.item()
            * batch_size
        )


        running_character_loss += (
            character_loss.item()
            * batch_size
        )


        running_source_loss += (
            source_loss.item()
            * batch_size
        )


        character_pred = (
            character_outputs
            .argmax(dim=1)
        )


        source_pred = (
            source_outputs
            .argmax(dim=1)
        )


        character_correct += (
            character_pred
            .eq(character_labels)
            .sum()
            .item()
        )


        source_correct += (
            source_pred
            .eq(source_labels)
            .sum()
            .item()
        )


        total += batch_size


        loop.set_postfix(

            char_acc=(
                f"{100.0 * character_correct / total:.2f}"
            ),

            source_acc=(
                f"{100.0 * source_correct / total:.2f}"
            ),

            source_w=(
                f"{SOURCE_WEIGHT:.2f}"
            )
        )


    return (

        running_total_loss / total,

        running_character_loss / total,

        running_source_loss / total,

        100.0
        * character_correct
        / total,

        100.0
        * source_correct
        / total
    )


# =========================================================
# 14. 验证
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()


    running_total_loss = 0.0
    running_character_loss = 0.0
    running_source_loss = 0.0


    character_top1_correct = 0
    character_top5_correct = 0

    source_correct = 0

    total = 0


    all_character_labels = []
    all_character_preds = []

    all_source_labels = []
    all_source_preds = []


    for (
        images,
        character_labels,
        source_labels

    ) in tqdm(
        loader,
        desc="Val"
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


        character_loss = (
            character_criterion(
                character_outputs,
                character_labels
            )
        )


        source_loss = (
            source_criterion(
                source_outputs,
                source_labels
            )
        )


        total_loss = (

            character_loss

            +

            SOURCE_WEIGHT
            * source_loss
        )


        batch_size = images.size(0)


        running_total_loss += (
            total_loss.item()
            * batch_size
        )


        running_character_loss += (
            character_loss.item()
            * batch_size
        )


        running_source_loss += (
            source_loss.item()
            * batch_size
        )


        # Character Top-1
        character_pred = (
            character_outputs
            .argmax(dim=1)
        )


        character_top1_correct += (
            character_pred
            .eq(character_labels)
            .sum()
            .item()
        )


        # Character Top-5
        _, character_top5 = (
            character_outputs.topk(
                5,
                dim=1,
                largest=True,
                sorted=True
            )
        )


        character_top5_correct += (

            character_top5.eq(

                character_labels
                .view(-1, 1)
                .expand_as(
                    character_top5
                )

            )
            .sum()
            .item()
        )


        # Source
        source_pred = (
            source_outputs
            .argmax(dim=1)
        )


        source_correct += (
            source_pred
            .eq(source_labels)
            .sum()
            .item()
        )


        total += batch_size


        all_character_labels.extend(
            character_labels
            .cpu()
            .numpy()
        )


        all_character_preds.extend(
            character_pred
            .cpu()
            .numpy()
        )


        all_source_labels.extend(
            source_labels
            .cpu()
            .numpy()
        )


        all_source_preds.extend(
            source_pred
            .cpu()
            .numpy()
        )


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
        all_character_preds,
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
        all_source_preds,
        average="macro",
        zero_division=0
    )


    return (

        running_total_loss / total,

        running_character_loss / total,

        running_source_loss / total,

        character_top1,

        character_top5,

        character_macro_f1,

        source_top1,

        source_macro_f1
    )


# =========================================================
# 15. 正式训练
# =========================================================

best_character_top1 = -1.0
best_epoch = 0

best_character_top5 = 0.0
best_character_macro_f1 = 0.0

best_source_top1 = 0.0
best_source_macro_f1 = 0.0

best_total_loss = 0.0
best_character_loss = 0.0
best_source_loss = 0.0


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
        train_total_loss,

        train_character_loss,

        train_source_loss,

        train_character_acc,

        train_source_acc

    ) = train_one_epoch(
        model,
        train_loader
    )


    (
        val_total_loss,

        val_character_loss,

        val_source_loss,

        val_character_top1,

        val_character_top5,

        val_character_macro_f1,

        val_source_top1,

        val_source_macro_f1

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


    selection_score_reference = (

        val_character_top1
        + val_source_top1

    ) / 2.0


    print()

    print(
        f"Train Total Loss: "
        f"{train_total_loss:.4f}"
    )

    print(
        f"Train Character Loss: "
        f"{train_character_loss:.4f}"
    )

    print(
        f"Train Source Loss: "
        f"{train_source_loss:.4f}"
    )


    print(
        f"Train Character Acc: "
        f"{train_character_acc:.2f}%"
    )

    print(
        f"Train Source Acc: "
        f"{train_source_acc:.2f}%"
    )


    print()

    print(
        f"Val Total Loss: "
        f"{val_total_loss:.4f}"
    )

    print(
        f"Val Character Loss: "
        f"{val_character_loss:.4f}"
    )

    print(
        f"Val Source Loss: "
        f"{val_source_loss:.4f}"
    )


    print()

    print(
        f"Val Character Top-1: "
        f"{val_character_top1:.2f}%"
    )

    print(
        f"Val Character Top-5: "
        f"{val_character_top5:.2f}%"
    )

    print(
        f"Val Character Macro-F1: "
        f"{val_character_macro_f1:.4f}"
    )


    print(
        f"Val Source Top-1: "
        f"{val_source_top1:.2f}%"
    )

    print(
        f"Val Source Macro-F1: "
        f"{val_source_macro_f1:.4f}"
    )


    print()

    print(
        "Character Weight: 1.0000"
    )

    print(
        f"Source Weight: "
        f"{SOURCE_WEIGHT:.4f}"
    )


    print(
        f"Selection Score (reference): "
        f"{selection_score_reference:.4f}"
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

        train_total_loss,

        train_character_loss,

        train_source_loss,

        train_character_acc,

        train_source_acc,

        val_total_loss,

        val_character_loss,

        val_source_loss,

        val_character_top1,

        val_character_top5,

        val_character_macro_f1,

        val_source_top1,

        val_source_macro_f1,

        1.0,

        SOURCE_WEIGHT,

        selection_score_reference,

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

            "train_total_loss",

            "train_character_loss",

            "train_source_loss",

            "train_character_acc",

            "train_source_acc",

            "val_total_loss",

            "val_character_loss",

            "val_source_loss",

            "val_character_top1",

            "val_character_top5",

            "val_character_macro_f1",

            "val_source_top1",

            "val_source_macro_f1",

            "character_weight",

            "source_weight",

            "selection_score_reference",

            "learning_rate",

            "epoch_time"
        ])


        writer.writerows(
            log_rows
        )


    # =====================================================
    # Character主任务选择最佳模型
    # =====================================================

    if (
        val_character_top1
        > best_character_top1
    ):

        best_character_top1 = (
            val_character_top1
        )

        best_epoch = epoch

        best_character_top5 = (
            val_character_top5
        )

        best_character_macro_f1 = (
            val_character_macro_f1
        )

        best_source_top1 = (
            val_source_top1
        )

        best_source_macro_f1 = (
            val_source_macro_f1
        )

        best_total_loss = (
            val_total_loss
        )

        best_character_loss = (
            val_character_loss
        )

        best_source_loss = (
            val_source_loss
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
            "保存新的最佳Character模型"
        )


# =========================================================
# 16. 保存摘要
# =========================================================

summary = {

    "experiment":
        EXPERIMENT_NAME,

    "source_weight":
        SOURCE_WEIGHT,

    "best_epoch":
        best_epoch,

    "val_character_top1":
        best_character_top1,

    "val_character_top5":
        best_character_top5,

    "val_character_macro_f1":
        best_character_macro_f1,

    "val_source_top1":
        best_source_top1,

    "val_source_macro_f1":
        best_source_macro_f1,

    "val_total_loss":
        best_total_loss,

    "val_character_loss":
        best_character_loss,

    "val_source_loss":
        best_source_loss,

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
# 17. 最终输出
# =========================================================

print()

print(
    "========== Auxiliary Task训练完成 =========="
)


print(
    "Source Loss Weight:",
    SOURCE_WEIGHT
)


print()

print(
    "最佳 Epoch:",
    best_epoch
)


print()

print(
    "Val Character Top-1:",
    best_character_top1
)

print(
    "Val Character Top-5:",
    best_character_top5
)

print(
    "Val Character Macro-F1:",
    best_character_macro_f1
)


print()

print(
    "Val Source Top-1:",
    best_source_top1
)

print(
    "Val Source Macro-F1:",
    best_source_macro_f1
)


print()

print(
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


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
