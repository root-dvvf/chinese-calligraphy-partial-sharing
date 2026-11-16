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

RESULT_DIR = (
    "/root/calligraphy_sci/results/"
    "resnet18_cb_beta099"
)

CHECKPOINT_DIR = (
    "/root/calligraphy_sci/checkpoints/"
    "resnet18_cb_beta099"
)

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
IMAGE_SIZE = 224

# 这次唯一主要改变的参数
CB_BETA = 0.99

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
# 4. 数据增强
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # 汉字不进行水平翻转
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
# 6. GPU
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
# 7. ResNet18
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(
    device
)


# =========================================================
# 8. Class-Balanced 权重计算
# =========================================================

def get_class_balanced_weights(
    train_csv,
    num_classes,
    beta
):

    df = pd.read_csv(
        train_csv
    )

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .reindex(
            range(num_classes),
            fill_value=0
        )
        .values
    )

    class_counts = np.asarray(
        class_counts,
        dtype=np.float64
    )

    # Effective Number of Samples
    effective_num = (
        1.0
        - np.power(
            beta,
            class_counts
        )
    )

    weights = (
        (1.0 - beta)
        / effective_num
    )

    # 归一化，使平均权重约为1
    weights = (
        weights
        / np.sum(weights)
        * num_classes
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    return (
        weights,
        class_counts
    )


print()
print(
    "正在计算 Class-Balanced 权重..."
)

print(
    "Beta:",
    CB_BETA
)


cb_weights, class_counts = (
    get_class_balanced_weights(
        TRAIN_CSV,
        NUM_CLASSES,
        CB_BETA
    )
)


print(
    "最少训练样本数:",
    int(class_counts.min())
)

print(
    "最多训练样本数:",
    int(class_counts.max())
)

print(
    "最小类别权重:",
    float(cb_weights.min())
)

print(
    "最大类别权重:",
    float(cb_weights.max())
)

print(
    "最大/最小权重比:",
    float(
        cb_weights.max()
        / cb_weights.min()
    )
)


# =========================================================
# 9. 保存类别权重
# =========================================================

weights_df = pd.DataFrame({

    "label":
        list(range(NUM_CLASSES)),

    "train_count":
        class_counts.astype(int),

    "cb_weight":
        cb_weights.numpy()
})


weights_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "class_balanced_weights.csv"
    ),

    index=False,
    encoding="utf-8-sig"
)


cb_weights = cb_weights.to(
    device
)


# =========================================================
# 10. 损失函数
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=cb_weights
)


# =========================================================
# 11. 优化器
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# =========================================================
# 12. 学习率调度
# =========================================================

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 13. 保存实验配置
# =========================================================

config = {

    "experiment":
        "ResNet18 + CB Loss beta=0.99",

    "model":
        "ResNet18",

    "pretrained":
        False,

    "loss":
        "Class-Balanced CrossEntropy",

    "cb_beta":
        CB_BETA,

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
        SEED
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
# 15. 验证
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
# 16. 正式训练
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


    # 用 Val Top-1 选择最佳模型
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
# 17. 保存日志
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
# 18. 训练结束
# =========================================================

print()

print(
    "========== 训练完成 =========="
)

print(
    "Beta:",
    CB_BETA
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
