import os
import json
import random
import csv
import time
import argparse
import copy

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
    default=0.10,
    help="Source auxiliary task loss weight"
)

parser.add_argument(
    "--share-until",
    type=str,
    default="layer2",
    choices=["layer2", "layer3"],
    help="共享到ResNet18哪一层"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


SOURCE_WEIGHT = args.source_weight
SHARE_UNTIL = args.share_until
SEED = args.seed


# =========================================================
# 2. 基础配置
# =========================================================

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

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224


# 例如：
# 0.10 -> 010
weight_tag = int(
    round(SOURCE_WEIGHT * 100)
)


EXPERIMENT_NAME = (
    f"partial_{SHARE_UNTIL}"
    f"_aux{weight_tag:03d}"
    f"_seed{SEED}"
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
    "共享到:",
    SHARE_UNTIL
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
# 5. 数据增强
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
# 7. Partial Sharing ResNet18
# =========================================================

class PartialSharingResNet18(nn.Module):

    def __init__(
        self,
        num_character_classes,
        num_source_classes,
        share_until="layer2"
    ):

        super().__init__()

        self.share_until = (
            share_until
        )


        # -------------------------------------------------
        # 创建标准ResNet18
        # -------------------------------------------------

        base = models.resnet18(
            weights=None
        )


        # =================================================
        # 所有模型都共享最前面：
        #
        # conv1
        # bn1
        # relu
        # maxpool
        # layer1
        # layer2
        # =================================================

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2


        # =================================================
        # 情况1：
        #
        # 共享到 layer2
        #
        # layer3 / layer4 分成两个任务分支
        # =================================================

        if share_until == "layer2":

            self.character_layer3 = (
                copy.deepcopy(
                    base.layer3
                )
            )

            self.character_layer4 = (
                copy.deepcopy(
                    base.layer4
                )
            )


            self.source_layer3 = (
                copy.deepcopy(
                    base.layer3
                )
            )

            self.source_layer4 = (
                copy.deepcopy(
                    base.layer4
                )
            )


        # =================================================
        # 情况2：
        #
        # 共享到 layer3
        #
        # 只有 layer4 分成两个任务分支
        # =================================================

        elif share_until == "layer3":

            self.layer3 = (
                base.layer3
            )


            self.character_layer4 = (
                copy.deepcopy(
                    base.layer4
                )
            )


            self.source_layer4 = (
                copy.deepcopy(
                    base.layer4
                )
            )


        else:

            raise ValueError(
                "share_until必须是layer2或layer3"
            )


        # =================================================
        # 池化
        # =================================================

        self.avgpool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # =================================================
        # 两个任务分类头
        #
        # ResNet18最后特征维度 = 512
        # =================================================

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

        # =================================================
        # Shared Stem
        # =================================================

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)


        x = self.layer1(x)

        x = self.layer2(x)


        # =================================================
        # Shared until layer2
        # =================================================

        if self.share_until == "layer2":

            # Character branch
            char_x = (
                self.character_layer3(
                    x
                )
            )

            char_x = (
                self.character_layer4(
                    char_x
                )
            )


            # Source branch
            source_x = (
                self.source_layer3(
                    x
                )
            )

            source_x = (
                self.source_layer4(
                    source_x
                )
            )


        # =================================================
        # Shared until layer3
        # =================================================

        else:

            x = self.layer3(x)


            char_x = (
                self.character_layer4(
                    x
                )
            )


            source_x = (
                self.source_layer4(
                    x
                )
            )


        # =================================================
        # Global Average Pooling
        # =================================================

        char_x = self.avgpool(
            char_x
        )

        source_x = self.avgpool(
            source_x
        )


        char_x = torch.flatten(
            char_x,
            1
        )

        source_x = torch.flatten(
            source_x,
            1
        )


        # =================================================
        # 两个分类头
        # =================================================

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
# 8. GPU
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print(
    "使用设备:",
    device
)


# =========================================================
# 9. 模型
# =========================================================

model = PartialSharingResNet18(

    NUM_CHARACTER_CLASSES,

    NUM_SOURCE_CLASSES,

    share_until=SHARE_UNTIL

).to(device)


# =========================================================
# 10. 参数量
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
    "模型总参数量:",
    total_params
)

print(
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 11. Loss
# =========================================================

character_criterion = (
    nn.CrossEntropyLoss()
)

source_criterion = (
    nn.CrossEntropyLoss()
)


# =========================================================
# 12. Optimizer
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
# 13. 保存配置
# =========================================================

config = {

    "experiment":
        EXPERIMENT_NAME,

    "architecture":
        "Partial Sharing ResNet18",

    "share_until":
        SHARE_UNTIL,

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
# 14. 单轮训练
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


        batch_size = (
            images.size(0)
        )


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
            character_outputs.argmax(
                dim=1
            )
        )


        source_pred = (
            source_outputs.argmax(
                dim=1
            )
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
# 15. 验证
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


        batch_size = (
            images.size(0)
        )


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


        # =================================================
        # Character Top-1
        # =================================================

        character_pred = (
            character_outputs.argmax(
                dim=1
            )
        )


        character_top1_correct += (
            character_pred
            .eq(character_labels)
            .sum()
            .item()
        )


        # =================================================
        # Character Top-5
        # =================================================

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


        # =================================================
        # Source
        # =================================================

        source_pred = (
            source_outputs.argmax(
                dim=1
            )
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
# 16. 正式训练
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
        "Shared Until:",
        SHARE_UNTIL
    )

    print(
        "Source Weight:",
        SOURCE_WEIGHT
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

        SOURCE_WEIGHT,

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

            "source_weight",

            "learning_rate",

            "epoch_time"
        ])


        writer.writerows(
            log_rows
        )


    # =====================================================
    # 仍然以 Character 为主任务选最佳模型
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
            "保存新的最佳Partial Sharing模型"
        )


# =========================================================
# 17. 保存摘要
# =========================================================

summary = {

    "experiment":
        EXPERIMENT_NAME,

    "architecture":
        "Partial Sharing ResNet18",

    "share_until":
        SHARE_UNTIL,

    "source_weight":
        SOURCE_WEIGHT,

    "seed":
        SEED,

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
# 18. 最终输出
# =========================================================

print()

print(
    "========== Partial Sharing训练完成 =========="
)

print(
    "共享到:",
    SHARE_UNTIL
)

print(
    "Source Weight:",
    SOURCE_WEIGHT
)

print(
    "Seed:",
    SEED
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
