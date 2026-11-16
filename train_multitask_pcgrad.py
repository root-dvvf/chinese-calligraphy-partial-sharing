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
# 1. 配置
# =========================================================

SEED = 42

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

TRAIN_CSV = "/root/calligraphy_sci/dual_splits/dual_train.csv"
VAL_CSV = "/root/calligraphy_sci/dual_splits/dual_val.csv"

RESULT_DIR = "/root/calligraphy_sci/results/multitask_pcgrad"
CHECKPOINT_DIR = "/root/calligraphy_sci/checkpoints/multitask_pcgrad"

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
# 3. Dataset
# =========================================================

class DualTaskDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = str(row["path"])

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
            image = self.transform(image)

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


    def forward(self, x):

        features = self.backbone(x)

        character_logits = (
            self.character_head(features)
        )

        source_logits = (
            self.source_head(features)
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
# 8. 模型
# =========================================================

model = MultiTaskResNet18(
    NUM_CHARACTER_CLASSES,
    NUM_SOURCE_CLASSES
).to(device)


# =========================================================
# 9. 参数划分
#
# Shared：
# ResNet18 Backbone
#
# Private：
# Character Head
# Source Head
# =========================================================

shared_params = [
    p
    for p in model.backbone.parameters()
    if p.requires_grad
]


character_head_params = [
    p
    for p in model.character_head.parameters()
    if p.requires_grad
]


source_head_params = [
    p
    for p in model.source_head.parameters()
    if p.requires_grad
]


# =========================================================
# 10. 参数量统计
# =========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


print()

print(
    "模型参数量:",
    total_params
)

print(
    "模型参数量(M):",
    f"{total_params / 1e6:.4f}"
)


# =========================================================
# 11. Loss
# =========================================================

character_criterion = nn.CrossEntropyLoss()
source_criterion = nn.CrossEntropyLoss()


# =========================================================
# 12. 优化器
#
# 与 Equal Weight 实验保持相同
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
# 13. 配置保存
# =========================================================

config = {

    "experiment":
        "MultiTask ResNet18 + PCGrad",

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

    "multitask_method":
        "PCGrad",

    "pcgrad_scope":
        "shared backbone parameters",

    "loss":
        "Character CE + Source CE",

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
# 14. PCGrad核心函数
# =========================================================

def apply_pcgrad(
    character_loss,
    source_loss
):

    """
    对共享 Backbone 的两个任务梯度做 PCGrad。

    如果：
        dot(g_char, g_source) < 0

    则移除冲突方向上的梯度分量。

    两个分类头仍然只接受各自任务的梯度。
    """

    # -----------------------------------------------------
    # 计算 Character 对共享 Backbone 的梯度
    # -----------------------------------------------------

    char_shared_grads = torch.autograd.grad(
        character_loss,
        shared_params,
        retain_graph=True
    )


    # -----------------------------------------------------
    # 计算 Source 对共享 Backbone 的梯度
    # -----------------------------------------------------

    source_shared_grads = torch.autograd.grad(
        source_loss,
        shared_params,
        retain_graph=True
    )


    # -----------------------------------------------------
    # 两个任务各自分类头的梯度
    # -----------------------------------------------------

    char_head_grads = torch.autograd.grad(
        character_loss,
        character_head_params,
        retain_graph=True
    )


    source_head_grads = torch.autograd.grad(
        source_loss,
        source_head_params
    )


    # -----------------------------------------------------
    # 整个共享Backbone上的 dot product
    # -----------------------------------------------------

    dot_product = torch.zeros(
        1,
        device=device
    )


    char_norm_sq = torch.zeros(
        1,
        device=device
    )


    source_norm_sq = torch.zeros(
        1,
        device=device
    )


    for gc, gs in zip(
        char_shared_grads,
        source_shared_grads
    ):

        dot_product += torch.sum(
            gc * gs
        )

        char_norm_sq += torch.sum(
            gc * gc
        )

        source_norm_sq += torch.sum(
            gs * gs
        )


    # -----------------------------------------------------
    # Cosine Similarity
    # -----------------------------------------------------

    denominator = torch.sqrt(
        char_norm_sq
        * source_norm_sq
    ) + 1e-12


    cosine_similarity = (
        dot_product
        / denominator
    ).item()


    conflict = (
        dot_product.item() < 0
    )


    # -----------------------------------------------------
    # PCGrad Projection
    # -----------------------------------------------------

    if conflict:

        char_coefficient = (
            dot_product
            / (source_norm_sq + 1e-12)
        )


        source_coefficient = (
            dot_product
            / (char_norm_sq + 1e-12)
        )


        projected_char_grads = []

        projected_source_grads = []


        for gc, gs in zip(
            char_shared_grads,
            source_shared_grads
        ):

            # Character梯度去掉与Source冲突部分
            new_gc = (
                gc
                - char_coefficient * gs
            )


            # Source梯度去掉与Character冲突部分
            new_gs = (
                gs
                - source_coefficient * gc
            )


            projected_char_grads.append(
                new_gc
            )

            projected_source_grads.append(
                new_gs
            )

    else:

        projected_char_grads = (
            char_shared_grads
        )

        projected_source_grads = (
            source_shared_grads
        )


    # -----------------------------------------------------
    # 清空旧梯度
    # -----------------------------------------------------

    optimizer.zero_grad()


    # -----------------------------------------------------
    # Shared Backbone
    #
    # 两个投影后的梯度相加
    # -----------------------------------------------------

    for (
        param,
        gc,
        gs

    ) in zip(

        shared_params,
        projected_char_grads,
        projected_source_grads

    ):

        param.grad = (
            gc + gs
        )


    # -----------------------------------------------------
    # Character Head
    # -----------------------------------------------------

    for (
        param,
        grad

    ) in zip(

        character_head_params,
        char_head_grads

    ):

        param.grad = grad


    # -----------------------------------------------------
    # Source Head
    # -----------------------------------------------------

    for (
        param,
        grad

    ) in zip(

        source_head_params,
        source_head_grads

    ):

        param.grad = grad


    return (
        conflict,
        cosine_similarity,
        math.sqrt(
            char_norm_sq.item()
        ),
        math.sqrt(
            source_norm_sq.item()
        )
    )


# =========================================================
# 15. 单轮训练
# =========================================================

def train_one_epoch(
    model,
    loader
):

    model.train()


    running_character_loss = 0.0
    running_source_loss = 0.0

    character_correct = 0
    source_correct = 0

    total = 0


    conflict_batches = 0
    total_batches = 0

    cosine_values = []

    char_grad_norms = []
    source_grad_norms = []


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


        # =================================================
        # 两个任务Loss
        # =================================================

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


        # =================================================
        # PCGrad
        # =================================================

        (
            conflict,
            cosine_similarity,
            char_grad_norm,
            source_grad_norm

        ) = apply_pcgrad(

            character_loss,
            source_loss
        )


        # 更新模型
        optimizer.step()


        # =================================================
        # 统计梯度关系
        # =================================================

        total_batches += 1


        if conflict:
            conflict_batches += 1


        cosine_values.append(
            cosine_similarity
        )


        char_grad_norms.append(
            char_grad_norm
        )


        source_grad_norms.append(
            source_grad_norm
        )


        # =================================================
        # Loss统计
        # =================================================

        batch_size = images.size(0)


        running_character_loss += (
            character_loss.item()
            * batch_size
        )


        running_source_loss += (
            source_loss.item()
            * batch_size
        )


        # =================================================
        # Accuracy
        # =================================================

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


        current_conflict_ratio = (
            100.0
            * conflict_batches
            / total_batches
        )


        loop.set_postfix(

            char_acc=(
                f"{100.0 * character_correct / total:.2f}"
            ),

            source_acc=(
                f"{100.0 * source_correct / total:.2f}"
            ),

            conflict=(
                f"{current_conflict_ratio:.1f}%"
            )
        )


    return (

        running_character_loss / total,

        running_source_loss / total,

        100.0
        * character_correct
        / total,

        100.0
        * source_correct
        / total,

        100.0
        * conflict_batches
        / total_batches,

        float(
            np.mean(
                cosine_values
            )
        ),

        float(
            np.mean(
                char_grad_norms
            )
        ),

        float(
            np.mean(
                source_grad_norms
            )
        )
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


        batch_size = images.size(0)


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
        # Source
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

        running_character_loss / total,

        running_source_loss / total,

        character_top1,

        character_top5,

        character_macro_f1,

        source_top1,

        source_macro_f1
    )


# =========================================================
# 17. 正式训练
# =========================================================

best_score = -1.0

best_epoch = 0

best_character_top1 = 0.0
best_character_macro_f1 = 0.0

best_source_top1 = 0.0
best_source_macro_f1 = 0.0

best_conflict_ratio = 0.0
best_mean_cosine = 0.0


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
        train_character_loss,

        train_source_loss,

        train_character_acc,

        train_source_acc,

        train_conflict_ratio,

        train_mean_cosine,

        train_character_grad_norm,

        train_source_grad_norm

    ) = train_one_epoch(

        model,
        train_loader
    )


    (
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


    # 与前面所有双任务实验保持一致
    selection_score = (

        val_character_top1
        + val_source_top1

    ) / 2.0


    print()

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
        f"Train Gradient Conflict Ratio: "
        f"{train_conflict_ratio:.2f}%"
    )

    print(
        f"Train Mean Cosine Similarity: "
        f"{train_mean_cosine:.6f}"
    )

    print(
        f"Train Character Grad Norm: "
        f"{train_character_grad_norm:.6f}"
    )

    print(
        f"Train Source Grad Norm: "
        f"{train_source_grad_norm:.6f}"
    )


    print()

    print(
        f"Val Character Loss: "
        f"{val_character_loss:.4f}"
    )

    print(
        f"Val Source Loss: "
        f"{val_source_loss:.4f}"
    )


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
    # 日志
    # =====================================================

    log_rows.append([

        epoch,

        train_character_loss,

        train_source_loss,

        train_character_acc,

        train_source_acc,

        train_conflict_ratio,

        train_mean_cosine,

        train_character_grad_norm,

        train_source_grad_norm,

        val_character_loss,

        val_source_loss,

        val_character_top1,

        val_character_top5,

        val_character_macro_f1,

        val_source_top1,

        val_source_macro_f1,

        selection_score,

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

            "train_character_loss",

            "train_source_loss",

            "train_character_acc",

            "train_source_acc",

            "train_conflict_ratio",

            "train_mean_cosine",

            "train_character_grad_norm",

            "train_source_grad_norm",

            "val_character_loss",

            "val_source_loss",

            "val_character_top1",

            "val_character_top5",

            "val_character_macro_f1",

            "val_source_top1",

            "val_source_macro_f1",

            "selection_score",

            "learning_rate",

            "epoch_time"
        ])


        writer.writerows(
            log_rows
        )


    # =====================================================
    # 最佳模型
    # =====================================================

    if selection_score > best_score:

        best_score = selection_score

        best_epoch = epoch

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

        best_conflict_ratio = (
            train_conflict_ratio
        )

        best_mean_cosine = (
            train_mean_cosine
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
            "保存新的最佳PCGrad模型"
        )


# =========================================================
# 18. 保存摘要
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

    "train_conflict_ratio":
        best_conflict_ratio,

    "train_mean_cosine_similarity":
        best_mean_cosine,

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
# 19. 最终结果
# =========================================================

print()

print(
    "========== PCGrad训练完成 =========="
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
    "最佳轮次训练梯度冲突比例:",
    best_conflict_ratio
)

print(
    "最佳轮次平均Cosine:",
    best_mean_cosine
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
