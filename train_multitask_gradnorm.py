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
    "multitask_gradnorm"
)

CHECKPOINT_DIR = (
    "/root/calligraphy_sci/checkpoints/"
    "multitask_gradnorm"
)

BATCH_SIZE = 128
NUM_EPOCHS = 20

# 模型本身学习率
LEARNING_RATE = 1e-3

# GradNorm任务权重学习率
WEIGHT_LEARNING_RATE = 0.025

# GradNorm不对称参数
GRADNORM_ALPHA = 1.5

NUM_WORKERS = 8
IMAGE_SIZE = 224

NUM_TASKS = 2

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

    # 与之前实验完全一致
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
# 5. DataLoader
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

        self.backbone = models.resnet18(
            weights=None
        )

        feature_dim = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        # 汉字分类头
        self.character_head = nn.Linear(
            feature_dim,
            num_character_classes
        )

        # 来源分类头
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
# 7. GPU
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
# 8. 创建模型
# =========================================================

model = MultiTaskResNet18(
    NUM_CHARACTER_CLASSES,
    NUM_SOURCE_CLASSES
)

model = model.to(
    device
)


# =========================================================
# 9. GradNorm任务权重
#
# 初始:
# Character = 1
# Source = 1
#
# 并始终归一化为：
#
# w_char + w_source = 2
# =========================================================

task_weights = nn.Parameter(

    torch.ones(
        NUM_TASKS,
        device=device
    )
)


# =========================================================
# 10. 参数量
# =========================================================

model_params = sum(
    p.numel()
    for p in model.parameters()
)

total_params = (
    model_params
    + task_weights.numel()
)

print()

print(
    "模型参数量:",
    model_params
)

print(
    "GradNorm额外参数:",
    task_weights.numel()
)

print(
    "总参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 11. 损失
# =========================================================

character_criterion = (
    nn.CrossEntropyLoss()
)

source_criterion = (
    nn.CrossEntropyLoss()
)


# =========================================================
# 12. 模型优化器
#
# 与Equal Weight实验一致
# =========================================================

model_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 13. GradNorm权重优化器
# =========================================================

weight_optimizer = torch.optim.Adam(
    [task_weights],
    lr=WEIGHT_LEARNING_RATE
)


# =========================================================
# 14. 模型学习率调度
# =========================================================

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        model_optimizer,
        T_max=NUM_EPOCHS
    )
)


# =========================================================
# 15. 配置保存
# =========================================================

config = {

    "experiment":
        "MultiTask ResNet18 + GradNorm",

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

    "weighting_method":
        "GradNorm",

    "gradnorm_alpha":
        GRADNORM_ALPHA,

    "initial_character_weight":
        1.0,

    "initial_source_weight":
        1.0,

    "weight_sum_constraint":
        2.0,

    "model_learning_rate":
        LEARNING_RATE,

    "weight_learning_rate":
        WEIGHT_LEARNING_RATE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        NUM_EPOCHS,

    "model_optimizer":
        "AdamW",

    "weight_optimizer":
        "Adam",

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
# 16. GradNorm初始Loss
#
# 第一次训练Batch时记录：
#
# L_char(0)
# L_source(0)
#
# 后续用来判断哪个任务学得快/慢
# =========================================================

initial_losses = None


# =========================================================
# 17. 单轮训练
# =========================================================

def train_one_epoch(
    model,
    loader
):

    global initial_losses

    model.train()

    running_weighted_loss = 0.0
    running_character_loss = 0.0
    running_source_loss = 0.0
    running_gradnorm_loss = 0.0

    character_correct = 0
    source_correct = 0
    total = 0

    # 用ResNet18最后一个共享卷积层
    # 测量两个任务对共享网络的梯度
    shared_parameter = (
        model.backbone
        .layer4[-1]
        .conv2
        .weight
    )


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


        # =================================================
        # Forward
        # =================================================

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


        losses = torch.stack([
            character_loss,
            source_loss
        ])


        # =================================================
        # 首Batch记录初始任务Loss
        # =================================================

        if initial_losses is None:

            initial_losses = (
                losses
                .detach()
                .clone()
            )

            print()

            print(
                "记录GradNorm初始Loss:"
            )

            print(
                "Initial Character Loss:",
                initial_losses[0].item()
            )

            print(
                "Initial Source Loss:",
                initial_losses[1].item()
            )


        # =================================================
        # 计算两个任务对最后共享层的原始梯度范数
        #
        # ∇W L_char
        # ∇W L_source
        # =================================================

        char_grad = torch.autograd.grad(
            character_loss,
            shared_parameter,
            retain_graph=True
        )[0]


        source_grad = torch.autograd.grad(
            source_loss,
            shared_parameter,
            retain_graph=True
        )[0]


        # 不需要对共享网络再求二阶梯度
        char_grad_norm_base = (
            char_grad
            .detach()
            .norm(2)
        )

        source_grad_norm_base = (
            source_grad
            .detach()
            .norm(2)
        )


        base_grad_norms = torch.stack([
            char_grad_norm_base,
            source_grad_norm_base
        ])


        # =================================================
        # 实际加权梯度范数
        #
        # G_i = w_i * ||∇ L_i||
        # =================================================

        grad_norms = (
            task_weights
            * base_grad_norms
        )


        # =================================================
        # 计算相对训练速度
        #
        # loss_ratio_i =
        # L_i(t) / L_i(0)
        #
        # r_i =
        # loss_ratio_i /
        # mean(loss_ratio)
        #
        # r > 1:
        # 这个任务学得相对慢
        #
        # r < 1:
        # 这个任务学得相对快
        # =================================================

        with torch.no_grad():

            loss_ratios = (
                losses.detach()
                / initial_losses
            )

            relative_training_rates = (
                loss_ratios
                / loss_ratios.mean()
            )


            mean_grad_norm = (
                grad_norms
                .detach()
                .mean()
            )


            target_grad_norms = (

                mean_grad_norm

                * torch.pow(
                    relative_training_rates,
                    GRADNORM_ALPHA
                )
            )


        # =================================================
        # GradNorm Loss
        #
        # 让实际梯度向目标梯度靠近
        # =================================================

        gradnorm_loss = torch.sum(

            torch.abs(
                grad_norms
                - target_grad_norms
            )
        )


        # =================================================
        # 先计算任务权重应该怎么更新
        # =================================================

        weight_grad = torch.autograd.grad(

            gradnorm_loss,

            task_weights,

            retain_graph=True

        )[0]


        # =================================================
        # 更新网络参数
        #
        # 注意：
        # 对网络而言，task_weights要detach
        #
        # GradNorm权重由自己的优化器更新
        # =================================================

        model_optimizer.zero_grad()


        weighted_loss = (

            task_weights[0].detach()
            * character_loss

            +

            task_weights[1].detach()
            * source_loss
        )


        weighted_loss.backward()

        model_optimizer.step()


        # =================================================
        # 更新GradNorm任务权重
        # =================================================

        weight_optimizer.zero_grad()

        task_weights.grad = (
            weight_grad
        )

        weight_optimizer.step()


        # =================================================
        # 保证权重：
        #
        # 1. 始终为正数
        # 2. 总和始终 = 2
        #
        # 防止Uncertainty实验那样无限膨胀
        # =================================================

        with torch.no_grad():

            task_weights.clamp_(
                min=1e-3
            )

            task_weights.mul_(

                NUM_TASKS
                / task_weights.sum()
            )


        # =================================================
        # 统计
        # =================================================

        batch_size = (
            images.size(0)
        )


        running_weighted_loss += (
            weighted_loss.item()
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


        running_gradnorm_loss += (
            gradnorm_loss.item()
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

            w_char=(
                f"{task_weights[0].item():.3f}"
            ),

            w_src=(
                f"{task_weights[1].item():.3f}"
            )
        )


    return (

        running_weighted_loss / total,

        running_character_loss / total,

        running_source_loss / total,

        running_gradnorm_loss / total,

        100.0
        * character_correct
        / total,

        100.0
        * source_correct
        / total,

        task_weights[0].item(),

        task_weights[1].item()
    )


# =========================================================
# 18. 验证
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    running_weighted_loss = 0.0

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


        weighted_loss = (

            task_weights[0]
            * character_loss

            +

            task_weights[1]
            * source_loss
        )


        batch_size = (
            images.size(0)
        )


        running_weighted_loss += (
            weighted_loss.item()
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
            character_outputs
            .argmax(dim=1)
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


        # =================================================
        # Source Top-1
        # =================================================

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

        running_weighted_loss / total,

        running_character_loss / total,

        running_source_loss / total,

        character_top1,

        character_top5,

        character_macro_f1,

        source_top1,

        source_macro_f1
    )


# =========================================================
# 19. 正式训练
# =========================================================

best_score = -1.0

best_epoch = 0

best_character_top1 = 0.0

best_character_macro_f1 = 0.0

best_source_top1 = 0.0

best_source_macro_f1 = 0.0

best_character_weight = 1.0

best_source_weight = 1.0


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
        train_weighted_loss,

        train_character_loss,

        train_source_loss,

        train_gradnorm_loss,

        train_character_acc,

        train_source_acc,

        character_weight,

        source_weight

    ) = train_one_epoch(

        model,

        train_loader
    )


    (
        val_weighted_loss,

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

        model_optimizer
        .param_groups[0]["lr"]
    )


    scheduler.step()


    elapsed = (
        time.time()
        - start_time
    )


    # 与之前双任务实验完全相同
    selection_score = (

        val_character_top1
        + val_source_top1

    ) / 2.0


    print()

    print(
        f"Train Weighted Loss: "
        f"{train_weighted_loss:.4f}"
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
        f"Train GradNorm Loss: "
        f"{train_gradnorm_loss:.4f}"
    )


    print()

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
        f"Val Weighted Loss: "
        f"{val_weighted_loss:.4f}"
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
        f"GradNorm Character Weight: "
        f"{character_weight:.6f}"
    )

    print(
        f"GradNorm Source Weight: "
        f"{source_weight:.6f}"
    )

    print(
        f"Weight Sum: "
        f"{character_weight + source_weight:.6f}"
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

        train_weighted_loss,

        train_character_loss,

        train_source_loss,

        train_gradnorm_loss,

        train_character_acc,

        train_source_acc,

        val_weighted_loss,

        val_character_loss,

        val_source_loss,

        val_character_top1,

        val_character_top5,

        val_character_macro_f1,

        val_source_top1,

        val_source_macro_f1,

        character_weight,

        source_weight,

        selection_score,

        current_lr,

        elapsed
    ])


    # 每轮保存，避免SSH断开损失日志
    with open(

        log_path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:


        writer = csv.writer(f)


        writer.writerow([

            "epoch",

            "train_weighted_loss",

            "train_character_loss",

            "train_source_loss",

            "train_gradnorm_loss",

            "train_character_acc",

            "train_source_acc",

            "val_weighted_loss",

            "val_character_loss",

            "val_source_loss",

            "val_character_top1",

            "val_character_top5",

            "val_character_macro_f1",

            "val_source_top1",

            "val_source_macro_f1",

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
    # 保存最佳模型
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


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "task_weights":
                    task_weights.detach().cpu(),

                "initial_losses":
                    initial_losses.detach().cpu(),

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
            "保存新的最佳GradNorm模型"
        )


# =========================================================
# 20. 保存最佳结果摘要
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

    "character_weight":
        best_character_weight,

    "source_weight":
        best_source_weight,

    "weight_sum":
        (
            best_character_weight
            + best_source_weight
        ),

    "gradnorm_alpha":
        GRADNORM_ALPHA,

    "weight_learning_rate":
        WEIGHT_LEARNING_RATE,

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
# 21. 最终结果
# =========================================================

print()

print(
    "========== GradNorm训练完成 =========="
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
    "最佳 Character Weight:",
    best_character_weight
)

print(
    "最佳 Source Weight:",
    best_source_weight
)

print(
    "Weight Sum:",
    best_character_weight
    + best_source_weight
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
