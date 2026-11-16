import os
import json
import time
import random
import copy
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# =====================================================
# 1. 参数
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--arch",
    type=str,
    required=True,
    choices=["full", "partial"]
)

parser.add_argument(
    "--seed",
    type=int,
    required=True
)

args = parser.parse_args()

ARCH = args.arch
SEED = args.seed

ROOT = "/root/calligraphy_sci"

NUM_CHAR = 6249
NUM_SOURCE = 19

SOURCE_WEIGHT = 1.0

EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224

if ARCH == "full":
    NAME = f"equal_fullshare_seed{SEED}"
else:
    NAME = f"equal_partial_layer3_seed{SEED}"

RESULT_DIR = os.path.join(
    ROOT, "results", NAME
)

CKPT_DIR = os.path.join(
    ROOT, "checkpoints", NAME
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(
    ROOT, "dual_splits", "dual_train.csv"
)

VAL_CSV = os.path.join(
    ROOT, "dual_splits", "dual_val.csv"
)


# =====================================================
# 2. 随机种子
# =====================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =====================================================
# 3. Dataset
# =====================================================

class DualDataset(Dataset):

    def __init__(self, csv_path, transform):

        df = pd.read_csv(csv_path)

        self.paths = df["path"].astype(str).tolist()

        self.char_labels = (
            df["character_label"].astype(int).tolist()
        )

        self.source_labels = (
            df["source_label"].astype(int).tolist()
        )

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return (
            image,
            self.char_labels[idx],
            self.source_labels[idx]
        )


# =====================================================
# 4. 数据预处理
# =====================================================

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

train_dataset = DualDataset(
    TRAIN_CSV, train_transform
)

val_dataset = DualDataset(
    VAL_CSV, val_transform
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


# =====================================================
# 5. 完全共享 ResNet18
# =====================================================

class FullSharingResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        self.backbone.fc = nn.Identity()

        self.character_head = nn.Linear(
            512, NUM_CHAR
        )

        self.source_head = nn.Linear(
            512, NUM_SOURCE
        )

    def forward(self, x):

        feature = self.backbone(x)

        char_output = self.character_head(feature)
        source_output = self.source_head(feature)

        return char_output, source_output


# =====================================================
# 6. 部分共享 ResNet18
#
# conv1-layer3 共享
# layer4 与分类头分别独立
# =====================================================

class PartialSharingResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.resnet18(
            weights=None
        )

        # Shared layers

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        # Task-specific layer4

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
            512, NUM_CHAR
        )

        self.source_head = nn.Linear(
            512, NUM_SOURCE
        )

    def forward(self, x):

        # Shared representation

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Character branch

        char_feature = self.character_layer4(x)
        char_feature = self.avgpool(char_feature)
        char_feature = torch.flatten(
            char_feature, 1
        )

        char_output = self.character_head(
            char_feature
        )

        # Source branch

        source_feature = self.source_layer4(x)
        source_feature = self.avgpool(source_feature)
        source_feature = torch.flatten(
            source_feature, 1
        )

        source_output = self.source_head(
            source_feature
        )

        return char_output, source_output


# =====================================================
# 7. 创建模型
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if ARCH == "full":
    model = FullSharingResNet18()
else:
    model = PartialSharingResNet18()

model = model.to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

char_criterion = nn.CrossEntropyLoss()
source_criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# =====================================================
# 8. 保存配置
# =====================================================

config = {
    "experiment": NAME,
    "architecture": ARCH,
    "seed": SEED,
    "source_weight": SOURCE_WEIGHT,
    "loss": "L_char + 1.0 * L_source",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "image_size": IMAGE_SIZE,
    "augmentation": "RandomRotation(5)",
    "normalization": "ImageNet mean/std",
    "pretrained": False,
    "model_selection": "Val Character Top-1",
    "train_csv": TRAIN_CSV,
    "val_csv": VAL_CSV,
    "total_params": total_params
}

with open(
    os.path.join(RESULT_DIR, "config.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4,
        ensure_ascii=False
    )


# =====================================================
# 9. 单轮训练
# =====================================================

def train_one_epoch():

    model.train()

    total = 0
    total_loss = 0.0

    char_correct = 0
    source_correct = 0

    loop = tqdm(
        train_loader,
        desc="Train"
    )

    for images, y_char, y_source in loop:

        images = images.to(
            device, non_blocking=True
        )

        y_char = y_char.to(
            device, non_blocking=True
        )

        y_source = y_source.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        out_char, out_source = model(images)

        loss_char = char_criterion(
            out_char, y_char
        )

        loss_source = source_criterion(
            out_source, y_source
        )

        # Equal Weight

        loss = loss_char + loss_source

        loss.backward()
        optimizer.step()

        n = images.size(0)

        total += n
        total_loss += loss.item() * n

        char_correct += (
            out_char.argmax(1) == y_char
        ).sum().item()

        source_correct += (
            out_source.argmax(1) == y_source
        ).sum().item()

    return {
        "train_loss": total_loss / total,
        "train_char_top1": char_correct / total * 100,
        "train_source_top1": source_correct / total * 100
    }


# =====================================================
# 10. 验证
# =====================================================

@torch.no_grad()
def validate():

    model.eval()

    total = 0

    char_correct = 0
    char_top5_correct = 0

    source_correct = 0
    source_top5_correct = 0

    char_true = []
    char_pred = []

    source_true = []
    source_pred = []

    loop = tqdm(
        val_loader,
        desc="Val"
    )

    for images, y_char, y_source in loop:

        images = images.to(
            device, non_blocking=True
        )

        y_char = y_char.to(
            device, non_blocking=True
        )

        y_source = y_source.to(
            device, non_blocking=True
        )

        out_char, out_source = model(images)

        pred_char = out_char.argmax(dim=1)
        pred_source = out_source.argmax(dim=1)

        n = images.size(0)
        total += n

        char_correct += (
            pred_char == y_char
        ).sum().item()

        source_correct += (
            pred_source == y_source
        ).sum().item()

        char_top5 = out_char.topk(
            5, dim=1
        ).indices

        source_top5 = out_source.topk(
            5, dim=1
        ).indices

        char_top5_correct += (
            char_top5 == y_char.unsqueeze(1)
        ).any(dim=1).sum().item()

        source_top5_correct += (
            source_top5 == y_source.unsqueeze(1)
        ).any(dim=1).sum().item()

        char_true.extend(
            y_char.cpu().tolist()
        )

        char_pred.extend(
            pred_char.cpu().tolist()
        )

        source_true.extend(
            y_source.cpu().tolist()
        )

        source_pred.extend(
            pred_source.cpu().tolist()
        )

    return {
        "val_char_top1":
            char_correct / total * 100,

        "val_char_top5":
            char_top5_correct / total * 100,

        "val_char_macro_f1":
            f1_score(
                char_true,
                char_pred,
                average="macro",
                zero_division=0,
                labels=list(range(NUM_CHAR))
            ),

        "val_source_top1":
            source_correct / total * 100,

        "val_source_top5":
            source_top5_correct / total * 100,

        "val_source_macro_f1":
            f1_score(
                source_true,
                source_pred,
                average="macro",
                zero_division=0,
                labels=list(range(NUM_SOURCE))
            )
    }


# =====================================================
# 11. 正式训练
# =====================================================

print()
print("=" * 55)
print("EXPERIMENT:", NAME)
print("ARCH:", ARCH)
print("SEED:", SEED)
print("SOURCE WEIGHT:", SOURCE_WEIGHT)
print("TRAIN SAMPLES:", len(train_dataset))
print("VAL SAMPLES:", len(val_dataset))
print("DEVICE:", device)
print("PARAMS(M):", round(total_params / 1e6, 4))
print("=" * 55)

best_char_top1 = -1.0
best_summary = None

history = []

for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    print()
    print(
        f"========== Epoch {epoch}/{EPOCHS} =========="
    )

    train_result = train_one_epoch()
    val_result = validate()

    current_lr = optimizer.param_groups[0]["lr"]

    scheduler.step()

    epoch_time = time.time() - start_time

    log = {
        "epoch": epoch,
        **train_result,
        **val_result,
        "learning_rate": current_lr,
        "epoch_time": epoch_time
    }

    history.append(log)

    # Save training log

    pd.DataFrame(history).to_csv(
        os.path.join(
            RESULT_DIR, "train_log.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"Train Character: "
        f"{train_result['train_char_top1']:.4f}%"
    )

    print(
        f"Train Source: "
        f"{train_result['train_source_top1']:.4f}%"
    )

    print()
    print(
        f"Val Character Top-1: "
        f"{val_result['val_char_top1']:.6f}%"
    )

    print(
        f"Val Character Top-5: "
        f"{val_result['val_char_top5']:.6f}%"
    )

    print(
        f"Val Character Macro-F1: "
        f"{val_result['val_char_macro_f1']:.6f}"
    )

    print(
        f"Val Source Top-1: "
        f"{val_result['val_source_top1']:.6f}%"
    )

    print(
        f"Val Source Top-5: "
        f"{val_result['val_source_top5']:.6f}%"
    )

    print(
        f"Val Source Macro-F1: "
        f"{val_result['val_source_macro_f1']:.6f}"
    )

    print(
        f"Learning Rate: {current_lr:.8f}"
    )

    print(
        f"Epoch Time: {epoch_time:.1f}s"
    )

    # =================================================
    # Best checkpoint:
    # 按 Character Validation Top-1 选优
    # =================================================

    if val_result["val_char_top1"] > best_char_top1:

        best_char_top1 = val_result[
            "val_char_top1"
        ]

        best_summary = {
            "experiment": NAME,
            "architecture": ARCH,
            "seed": SEED,
            "best_epoch": epoch,
            "source_weight": SOURCE_WEIGHT,
            "total_params": total_params,
            **val_result
        }

        torch.save(
            model.state_dict(),
            os.path.join(
                CKPT_DIR, "best_model.pth"
            )
        )

        with open(
            os.path.join(
                RESULT_DIR, "best_summary.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                best_summary,
                f,
                indent=4,
                ensure_ascii=False
            )

        print()
        print("保存新的最佳模型")


# =====================================================
# 12. 最终输出
# =====================================================

print()
print("=" * 55)
print("EQUAL WEIGHT TRAINING COMPLETE")
print("=" * 55)

print("Experiment:", NAME)
print("Seed:", SEED)
print("Best Epoch:", best_summary["best_epoch"])

print()
print(
    "Val Character Top-1:",
    best_summary["val_char_top1"]
)

print(
    "Val Character Top-5:",
    best_summary["val_char_top5"]
)

print(
    "Val Character Macro-F1:",
    best_summary["val_char_macro_f1"]
)

print(
    "Val Source Top-1:",
    best_summary["val_source_top1"]
)

print(
    "Val Source Top-5:",
    best_summary["val_source_top5"]
)

print(
    "Val Source Macro-F1:",
    best_summary["val_source_macro_f1"]
)

print()
print("Results:", RESULT_DIR)
print("Checkpoints:", CKPT_DIR)
