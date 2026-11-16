import os
import json
import random
import csv
import time

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

RESULT_DIR = "/root/calligraphy_sci/results/resnet18_scratch_dedup"
CHECKPOINT_DIR = "/root/calligraphy_sci/checkpoints/resnet18_scratch_dedup"

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================
# 2. 固定随机种子
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. 数据集定义
# =========================================================

class CalligraphyDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.iloc[idx]["path"]
        label = int(self.data.iloc[idx]["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# 4. 图像预处理
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 5. DataLoader
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
# 6. 模型
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备:", device)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(device)


# =========================================================
# 7. 损失函数和优化器
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# =========================================================
# 8. 保存实验配置
# =========================================================

config = {
    "model": "ResNet18",
    "loss": "CrossEntropyLoss",
    "num_classes": NUM_CLASSES,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "image_size": IMAGE_SIZE,
    "seed": SEED
}

with open(
    os.path.join(RESULT_DIR, "config.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


# =========================================================
# 9. 训练函数
# =========================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Train")

    for images, labels in loop:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(
            loss=loss.item(),
            acc=100.0 * correct / total
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# =========================================================
# 10. 验证函数
# =========================================================

@torch.no_grad()
def validate(model, loader):

    model.eval()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Val")

    for images, labels in loop:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, pred_top1 = outputs.max(1)

        correct_top1 += pred_top1.eq(labels).sum().item()

        _, pred_top5 = outputs.topk(
            5,
            dim=1,
            largest=True,
            sorted=True
        )

        correct_top5 += (
            pred_top5.eq(
                labels.view(-1, 1).expand_as(pred_top5)
            )
            .sum()
            .item()
        )

        total += labels.size(0)

        all_preds.extend(
            pred_top1.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    val_loss = running_loss / total

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return val_loss, top1_acc, top5_acc, macro_f1


# =========================================================
# 11. 正式训练
# =========================================================

best_top1 = 0.0

log_rows = []

for epoch in range(1, NUM_EPOCHS + 1):

    print(
        f"\n========== Epoch {epoch}/{NUM_EPOCHS} =========="
    )

    start_time = time.time()

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_top1, val_top5, val_macro_f1 = validate(
        model,
        val_loader
    )

    scheduler.step()

    elapsed = time.time() - start_time

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc: {train_acc:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Top-1: {val_top1:.2f}%"
    )

    print(
        f"Val Top-5: {val_top5:.2f}%"
    )

    print(
        f"Val Macro-F1: {val_macro_f1:.4f}"
    )

    print(
        f"Epoch Time: {elapsed:.1f}s"
    )

    log_rows.append([
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_top1,
        val_top5,
        val_macro_f1,
        elapsed
    ])

    # 保存当前最好模型
    if val_top1 > best_top1:

        best_top1 = val_top1

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print("保存新的最佳模型")


# =========================================================
# 12. 保存训练日志
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
        "epoch_time"
    ])

    writer.writerows(log_rows)

print("\n训练完成")
print("最佳 Val Top-1:", best_top1)
print("训练日志:", log_path)
print(
    "最佳模型:",
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )
)
