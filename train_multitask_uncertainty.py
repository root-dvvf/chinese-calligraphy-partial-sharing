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

RESULT_DIR = (
    "/root/calligraphy_sci/results/"
    "multitask_uncertainty"
)

CHECKPOINT_DIR = (
    "/root/calligraphy_sci/checkpoints/"
    "multitask_uncertainty"
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
# 4. 图像预处理
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # 与之前所有正式实验保持一致
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
# 6. 双任务 ResNet18
# =========================================================

class MultiTaskResNet18(nn.Module):

    def __init__(
        self,
        num_character_classes,
        num_source_classes
    ):

        super().__init__()


        # ---------------------------------------------
        # 共享 ResNet18 Backbone
        # ---------------------------------------------

        self.backbone = models.resnet18(
            weights=None
        )


        feature_dim = (
            self.backbone.fc.in_features
        )


        # 删除原分类层
        self.backbone.fc = nn.Identity()


        # ---------------------------------------------
        # 汉字分类头
        # ---------------------------------------------

        self.character_head = nn.Linear(

            feature_dim,

            num_character_classes
        )


        # ---------------------------------------------
        # 19类书法来源分类头
        # ---------------------------------------------

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
# 7. Uncertainty Weighting
# =========================================================

class UncertaintyWeighting(nn.Module):

    def __init__(self):

        super().__init__()


        # 初始值全部为0
        #
        # exp(-0) = 1
        #
        # 所以训练刚开始时相当于：
        #
        # L = L_char + L_source
        #
        # 然后再由网络自动学习权重

        self.log_var_character = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

        self.log_var_source = (
            nn.Parameter(
                torch.zeros(1)
            )
        )


    def forward(
        self,
        character_loss,
        source_loss
    ):

        # ---------------------------------------------
        # 有效任务权重：
        #
        # weight = exp(-log_var)
        # ---------------------------------------------

        character_weight = torch.exp(
            -self.log_var_character
        )

        source_weight = torch.exp(
            -self.log_var_source
        )


        # ---------------------------------------------
        # 不确定性加权总损失
        #
        # L =
        # exp(-s1)*L1 + s1
        # +
        # exp(-s2)*L2 + s2
        # ---------------------------------------------

        total_loss = (

            character_weight
            * character_loss

            + self.log_var_character

            + source_weight
            * source_loss

            + self.log_var_source
        )


        return (

            total_loss,

            character_weight,

            source_weight
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


# =========================================================
# 9. 创建模型
# =========================================================

model = MultiTaskResNet18(

    NUM_CHARACTER_CLASSES,

    NUM_SOURCE_CLASSES
)

model = model.to(
    device
)


# =========================================================
# 10. 创建动态任务加权模块
# =========================================================

task_weighter = UncertaintyWeighting()

task_weighter = task_weighter.to(
    device
)


# =========================================================
# 11. 参数量统计
# =========================================================

model_params = sum(

    p.numel()

    for p in model.parameters()
)


weighting_params = sum(

    p.numel()

    for p in task_weighter.parameters()
)


total_params = (
    model_params
    + weighting_params
)


print()

print(
    "模型参数量:",
    model_params
)

print(
    "动态权重额外参数量:",
    weighting_params
)

print(
    "总参数量:",
    total_params
)

print(
    "总参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 12. 两个任务的 CE Loss
# =========================================================

character_criterion = (
    nn.CrossEntropyLoss()
)

source_criterion = (
    nn.CrossEntropyLoss()
)


# =========================================================
# 13. 优化器
#
# 注意：
# 必须同时优化模型参数和两个 log_var
# =========================================================

optimizer = torch.optim.AdamW(

    list(
        model.parameters()
    )
    +
    list(
        task_weighter.parameters()
    ),

    lr=LEARNING_RATE
)


# =========================================================
# 14. 学习率调度
# =========================================================

scheduler = (

    torch.optim.lr_scheduler
    .CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 15. 保存配置
# =========================================================

config = {

    "experiment":
        "MultiTask ResNet18 + Uncertainty Weighting",

    "backbone":
        "ResNet18",

    "pretrained":
        False,

    "task_1":
        "Character Recognition",

    "task_1_classes":
        NUM_CHARACTER_CLASSES,

    "task_2":
        "Calligraphic Source Classification",

    "task_2_classes":
        NUM_SOURCE_CLASSES,

    "loss_weighting":
        "Homoscedastic Uncertainty Weighting",

    "initial_log_var_character":
        0.0,

    "initial_log_var_source":
        0.0,

    "initial_character_weight":
        1.0,

    "initial_source_weight":
        1.0,

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
        total_params
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
# 16. 单轮训练
# =========================================================

def train_one_epoch(
    model,
    task_weighter,
    loader
):

    model.train()

    task_weighter.train()


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


        # ---------------------------------------------
        # 两个原始任务Loss
        # ---------------------------------------------

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


        # ---------------------------------------------
        # 动态不确定性任务加权
        # ---------------------------------------------

        (
            total_loss,
            character_weight,
            source_weight

        ) = task_weighter(

            character_loss,

            source_loss
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


        # ---------------------------------------------
        # Character Accuracy
        # ---------------------------------------------

        character_pred = (

            character_outputs
            .argmax(dim=1)
        )


        character_correct += (

            character_pred
            .eq(character_labels)
            .sum()
            .item()
        )


        # ---------------------------------------------
        # Source Accuracy
        # ---------------------------------------------

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


        loop.set_postfix(

            char_acc=(
                f"{100.0 * character_correct / total:.2f}"
            ),

            source_acc=(
                f"{100.0 * source_correct / total:.2f}"
            ),

            w_char=(
                f"{character_weight.item():.3f}"
            ),

            w_src=(
                f"{source_weight.item():.3f}"
            )
        )


    # -----------------------------------------------------
    # 当前Epoch结束时的动态权重
    # -----------------------------------------------------

    character_weight = torch.exp(

        -task_weighter
        .log_var_character

    ).item()


    source_weight = torch.exp(

        -task_weighter
        .log_var_source

    ).item()


    return (

        running_total_loss / total,

        running_character_loss / total,

        running_source_loss / total,

        100.0
        * character_correct
        / total,

        100.0
        * source_correct
        / total,

        task_weighter
        .log_var_character
        .item(),

        task_weighter
        .log_var_source
        .item(),

        character_weight,

        source_weight
    )


# =========================================================
# 17. 验证
# =========================================================

@torch.no_grad()
def validate(
    model,
    task_weighter,
    loader
):

    model.eval()

    task_weighter.eval()


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


        (
            total_loss,
            _,
            _

        ) = task_weighter(

            character_loss,

            source_loss
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


        # ---------------------------------------------
        # Character Top-1
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Character Top-5
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Source Top-1
        # ---------------------------------------------

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


    # -----------------------------------------------------
    # Character Metrics
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Source Metrics
    # -----------------------------------------------------

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
# 18. 正式训练
# =========================================================

best_score = -1.0

best_epoch = 0

best_character_top1 = 0.0

best_character_macro_f1 = 0.0

best_source_top1 = 0.0

best_source_macro_f1 = 0.0

best_character_weight = 1.0

best_source_weight = 1.0

best_log_var_character = 0.0

best_log_var_source = 0.0


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

        train_source_acc,

        log_var_character,

        log_var_source,

        character_weight,

        source_weight

    ) = train_one_epoch(

        model,

        task_weighter,

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

        task_weighter,

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


    # -----------------------------------------------------
    # 与 Equal Weight Baseline 完全相同的模型选择方式
    #
    # 两个任务 Top-1 平均
    # -----------------------------------------------------

    selection_score = (

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
        f"log_var Character: "
        f"{log_var_character:.6f}"
    )

    print(
        f"log_var Source: "
        f"{log_var_source:.6f}"
    )


    print(
        f"Effective Character Weight: "
        f"{character_weight:.6f}"
    )

    print(
        f"Effective Source Weight: "
        f"{source_weight:.6f}"
    )


    print()

    print(
        f"Selection Score: "
        f"{selection_score:.4f}"
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

        log_var_character,

        log_var_source,

        character_weight,

        source_weight,

        selection_score,

        current_lr,

        elapsed
    ])


    # 每轮写入文件
    # 防止 SSH 断线造成日志丢失

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

            "log_var_character",

            "log_var_source",

            "character_weight",

            "source_weight",

            "selection_score",

            "learning_rate",

            "epoch_time"
        ])


        writer.writerows(
            log_rows
        )


    # =====================================================
    # 保存综合最佳模型
    # =====================================================

    if selection_score > best_score:

        best_score = (
            selection_score
        )

        best_epoch = (
            epoch
        )

        best_character_top1 = (
            val_character_top1
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

        best_character_weight = (
            character_weight
        )

        best_source_weight = (
            source_weight
        )

        best_log_var_character = (
            log_var_character
        )

        best_log_var_source = (
            log_var_source
        )


        # ---------------------------------------------
        # 这里保存的不只是模型
        # 还保存动态权重参数
        # ---------------------------------------------

        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "weighter_state_dict":
                    task_weighter.state_dict(),

                "epoch":
                    epoch,

                "selection_score":
                    selection_score

            },

            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )


        print()

        print(
            "保存新的最佳不确定性加权模型"
        )


# =========================================================
# 19. 保存最佳结果摘要
# =========================================================

summary = {

    "best_epoch":
        best_epoch,

    "best_selection_score":
        best_score,

    "val_character_top1":
        best_character_top1,

    "val_character_macro_f1":
        best_character_macro_f1,

    "val_source_top1":
        best_source_top1,

    "val_source_macro_f1":
        best_source_macro_f1,

    "log_var_character":
        best_log_var_character,

    "log_var_source":
        best_log_var_source,

    "effective_character_weight":
        best_character_weight,

    "effective_source_weight":
        best_source_weight,

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
# 20. 最终结果
# =========================================================

print()

print(
    "========== 不确定性加权训练完成 =========="
)


print(
    "最佳 Epoch:",
    best_epoch
)

print(
    "最佳 Selection Score:",
    best_score
)


print()

print(
    "Val Character Top-1:",
    best_character_top1
)

print(
    "Val Character Macro-F1:",
    best_character_macro_f1
)


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
    "最终 Character log_var:",
    best_log_var_character
)

print(
    "最终 Source log_var:",
    best_log_var_source
)


print(
    "有效 Character Weight:",
    best_character_weight
)

print(
    "有效 Source Weight:",
    best_source_weight
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
