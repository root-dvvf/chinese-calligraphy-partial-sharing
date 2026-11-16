import os
import json
import random
import csv
import time
import math

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
# 1. 基本配置
# =========================================================

SEED = 42
NUM_CLASSES = 6249

TRAIN_CSV = "/root/calligraphy_sci/splits/train.csv"
VAL_CSV = "/root/calligraphy_sci/splits/val.csv"

RESULT_DIR = (
    "/root/calligraphy_sci/results/"
    "resnet18_eca"
)

CHECKPOINT_DIR = (
    "/root/calligraphy_sci/checkpoints/"
    "resnet18_eca"
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


# =========================================================
# 2. 固定随机种子
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. Dataset
# =========================================================

class CalligraphyDataset(Dataset):

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

        img_path = (
            self.data.iloc[idx]["path"]
        )

        label = int(
            self.data.iloc[idx]["label"]
        )

        image = Image.open(
            img_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# =========================================================
# 4. 图像预处理
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # 汉字不水平翻转，只进行小幅旋转
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
# 5. Dataset / DataLoader
# =========================================================

train_dataset = CalligraphyDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = CalligraphyDataset(
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
# 6. ECA 模块
# =========================================================

class ECABlock(nn.Module):

    def __init__(
        self,
        channels,
        gamma=2,
        b=1
    ):

        super().__init__()

        # 根据通道数量自动确定1D卷积核大小
        t = int(
            abs(
                (
                    math.log2(channels)
                    + b
                )
                / gamma
            )
        )

        kernel_size = (
            t
            if t % 2
            else t + 1
        )

        kernel_size = max(
            kernel_size,
            3
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(
                kernel_size - 1
            ) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        x
    ):

        # x:
        # [B, C, H, W]

        y = self.avg_pool(x)

        # [B, C, 1, 1]
        # -> [B, 1, C]

        y = (
            y.squeeze(-1)
            .transpose(-1, -2)
        )

        y = self.conv(y)

        # -> [B, C, 1]
        y = (
            y.transpose(-1, -2)
            .unsqueeze(-1)
        )

        y = self.sigmoid(y)

        return x * y.expand_as(x)


# =========================================================
# 7. 构建 ResNet18 + ECA
# =========================================================

def build_eca_resnet18(
    num_classes
):

    model = models.resnet18(
        weights=None
    )

    # 在ResNet18四个主要Stage之后加入ECA
    model.layer1 = nn.Sequential(
        model.layer1,
        ECABlock(64)
    )

    model.layer2 = nn.Sequential(
        model.layer2,
        ECABlock(128)
    )

    model.layer3 = nn.Sequential(
        model.layer3,
        ECABlock(256)
    )

    model.layer4 = nn.Sequential(
        model.layer4,
        ECABlock(512)
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


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


# =========================================================
# 9. 创建模型
# =========================================================

model = build_eca_resnet18(
    NUM_CLASSES
)

model = model.to(
    device
)


# =========================================================
# 10. 参数量统计
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

print()
print(
    "模型总参数量:",
    total_params
)

print(
    "可训练参数量:",
    trainable_params
)

print(
    "模型总参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 11. 损失函数
# =========================================================

criterion = (
    nn.CrossEntropyLoss()
)


# =========================================================
# 12. 优化器
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 13. 学习率调度
# =========================================================

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 14. 保存实验配置
# =========================================================

config = {

    "experiment":
        "ResNet18 + ECA + CE",

    "model":
        "ResNet18",

    "attention":
        "ECA",

    "eca_position":
        "after layer1-layer4",

    "pretrained":
        False,

    "loss":
        "CrossEntropyLoss",

    "num_classes":
        NUM_CLASSES,

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
# 15. 单轮训练
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


    for images, labels in loop:

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


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = (
            outputs.max(1)
        )


        total += (
            labels.size(0)
        )


        correct += (
            predicted
            .eq(labels)
            .sum()
            .item()
        )


        loop.set_postfix(

            loss=(
                f"{loss.item():.4f}"
            ),

            acc=(
                f"{100.0 * correct / total:.2f}"
            )
        )


    epoch_loss = (
        running_loss
        / total
    )


    epoch_acc = (
        100.0
        * correct
        / total
    )


    return (
        epoch_loss,
        epoch_acc
    )


# =========================================================
# 16. 验证
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    running_loss = 0.0

    correct_top1 = 0
    correct_top5 = 0

    total = 0

    all_preds = []
    all_labels = []


    loop = tqdm(
        loader,
        desc="Val"
    )


    for images, labels in loop:

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


        running_loss += (
            loss.item()
            * images.size(0)
        )


        # Top-1
        _, pred_top1 = (
            outputs.max(1)
        )


        correct_top1 += (
            pred_top1
            .eq(labels)
            .sum()
            .item()
        )


        # Top-5
        _, pred_top5 = outputs.topk(
            5,
            dim=1,
            largest=True,
            sorted=True
        )


        correct_top5 += (

            pred_top5.eq(

                labels
                .view(-1, 1)
                .expand_as(
                    pred_top5
                )

            )
            .sum()
            .item()
        )


        total += (
            labels.size(0)
        )


        all_preds.extend(
            pred_top1
            .cpu()
            .numpy()
        )


        all_labels.extend(
            labels
            .cpu()
            .numpy()
        )


    val_loss = (
        running_loss
        / total
    )


    top1_acc = (
        100.0
        * correct_top1
        / total
    )


    top5_acc = (
        100.0
        * correct_top5
        / total
    )


    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )


    return (
        val_loss,
        top1_acc,
        top5_acc,
        macro_f1
    )


# =========================================================
# 17. 正式训练
# =========================================================

best_top1 = 0.0
best_macro_f1 = 0.0
best_epoch = 0

log_rows = []


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


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc: "
        f"{train_acc:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Top-1: "
        f"{val_top1:.2f}%"
    )

    print(
        f"Val Top-5: "
        f"{val_top5:.2f}%"
    )

    print(
        f"Val Macro-F1: "
        f"{val_macro_f1:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{elapsed:.1f}s"
    )


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


    # =====================================================
    # 保存验证集 Top-1 最佳模型
    # =====================================================

    if val_top1 > best_top1:

        best_top1 = val_top1
        best_macro_f1 = val_macro_f1
        best_epoch = epoch


        torch.save(

            model.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )


        print(
            "保存新的最佳模型"
        )


# =========================================================
# 18. 保存训练日志
# =========================================================

log_path = os.path.join(
    RESULT_DIR,
    "train_log.csv"
)


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

        "train_acc",

        "val_loss",

        "val_top1",

        "val_top5",

        "val_macro_f1",

        "learning_rate",

        "epoch_time"

    ])


    writer.writerows(
        log_rows
    )


# =========================================================
# 19. 训练结束
# =========================================================

print()

print(
    "========== 训练完成 =========="
)

print(
    "最佳 Epoch:",
    best_epoch
)

print(
    "最佳 Val Top-1:",
    best_top1
)

print(
    "对应 Val Macro-F1:",
    best_macro_f1
)

print(
    "模型参数量:",
    total_params
)

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
