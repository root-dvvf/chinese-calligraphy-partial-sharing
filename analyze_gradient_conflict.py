import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from tqdm import tqdm


# =========================================================
# 1. 配置
# =========================================================

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

VAL_CSV = (
    "/root/calligraphy_sci/"
    "dual_splits/dual_val.csv"
)

MODEL_PATH = (
    "/root/calligraphy_sci/checkpoints/"
    "multitask_equal/best_model.pth"
)

OUT_DIR = (
    "/root/calligraphy_sci/results/"
    "gradient_conflict_analysis"
)

BATCH_SIZE = 128
NUM_WORKERS = 8
IMAGE_SIZE = 224

# 不需要分析全部108个batch
# 先随机/顺序分析50个已经足够观察趋势
MAX_BATCHES = 50

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =========================================================
# 2. Dataset
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

        image = Image.open(
            str(row["path"])
        ).convert("RGB")

        if self.transform:
            image = self.transform(
                image
            )

        character_label = int(
            row["character_label"]
        )

        source_label = int(
            row["source_label"]
        )

        return (
            image,
            character_label,
            source_label
        )


# =========================================================
# 3. 验证集预处理
# =========================================================

transform = transforms.Compose([

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


dataset = DualTaskDataset(
    VAL_CSV,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# =========================================================
# 4. 模型
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

        self.character_head = nn.Linear(
            feature_dim,
            num_character_classes
        )

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
# 5. GPU
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
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    state_dict
)

model = model.to(
    device
)

model.eval()


# =========================================================
# 6. 使用最后共享层进行梯度冲突分析
#
# GradNorm论文也通常选择最后共享层，
# 这样计算量较低。
# =========================================================

shared_parameter = (
    model.backbone
    .layer4[-1]
    .conv2
    .weight
)


character_criterion = (
    nn.CrossEntropyLoss()
)

source_criterion = (
    nn.CrossEntropyLoss()
)


# =========================================================
# 7. 分析每个Batch的梯度
# =========================================================

rows = []


for batch_idx, (
    images,
    character_labels,
    source_labels

) in enumerate(
    tqdm(
        loader,
        desc="Gradient Analysis"
    )
):

    if batch_idx >= MAX_BATCHES:
        break


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


    # =====================================================
    # Character梯度
    # =====================================================

    character_grad = torch.autograd.grad(
        character_loss,
        shared_parameter,
        retain_graph=True
    )[0]


    # =====================================================
    # Source梯度
    # =====================================================

    source_grad = torch.autograd.grad(
        source_loss,
        shared_parameter
    )[0]


    char_flat = (
        character_grad
        .detach()
        .flatten()
    )

    source_flat = (
        source_grad
        .detach()
        .flatten()
    )


    # =====================================================
    # 梯度范数
    # =====================================================

    char_norm = (
        torch.norm(
            char_flat,
            p=2
        )
        .item()
    )

    source_norm = (
        torch.norm(
            source_flat,
            p=2
        )
        .item()
    )


    # =====================================================
    # Cosine Similarity
    #
    # > 0：总体同方向
    # = 0：近似正交
    # < 0：存在梯度冲突
    # =====================================================

    cosine = (
        F.cosine_similarity(
            char_flat.unsqueeze(0),
            source_flat.unsqueeze(0),
            dim=1
        )
        .item()
    )


    dot_product = (
        torch.dot(
            char_flat,
            source_flat
        )
        .item()
    )


    conflict = (
        cosine < 0
    )


    rows.append({

        "batch":
            batch_idx + 1,

        "character_loss":
            character_loss.item(),

        "source_loss":
            source_loss.item(),

        "character_grad_norm":
            char_norm,

        "source_grad_norm":
            source_norm,

        "grad_norm_ratio_char_source":
            (
                char_norm
                / source_norm
                if source_norm > 0
                else np.nan
            ),

        "cosine_similarity":
            cosine,

        "dot_product":
            dot_product,

        "conflict":
            int(conflict)
    })


# =========================================================
# 8. 保存详细结果
# =========================================================

result_df = pd.DataFrame(
    rows
)


detail_path = os.path.join(
    OUT_DIR,
    "gradient_conflict_batches.csv"
)


result_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. 汇总
# =========================================================

total_batches = len(
    result_df
)

conflict_batches = int(
    result_df[
        "conflict"
    ].sum()
)

conflict_ratio = (
    conflict_batches
    / total_batches
    * 100.0
)


mean_cosine = (
    result_df[
        "cosine_similarity"
    ].mean()
)

median_cosine = (
    result_df[
        "cosine_similarity"
    ].median()
)


mean_char_norm = (
    result_df[
        "character_grad_norm"
    ].mean()
)

mean_source_norm = (
    result_df[
        "source_grad_norm"
    ].mean()
)


print()

print(
    "========== Gradient Conflict Result =========="
)

print(
    "分析Batch数:",
    total_batches
)

print(
    "梯度冲突Batch数:",
    conflict_batches
)

print(
    f"梯度冲突比例: "
    f"{conflict_ratio:.2f}%"
)


print()

print(
    f"平均 Cosine Similarity: "
    f"{mean_cosine:.6f}"
)

print(
    f"中位数 Cosine Similarity: "
    f"{median_cosine:.6f}"
)


print()

print(
    f"平均 Character Gradient Norm: "
    f"{mean_char_norm:.6f}"
)

print(
    f"平均 Source Gradient Norm: "
    f"{mean_source_norm:.6f}"
)


if mean_source_norm > 0:

    print(
        "平均梯度范数 Character / Source:",
        mean_char_norm
        / mean_source_norm
    )


print()

print(
    "结果保存:"
)

print(
    detail_path
)
