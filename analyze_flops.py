import copy

import torch
import torch.nn as nn
from torchvision import models

from thop import profile


# =========================================================
# 1. 配置
# =========================================================

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

IMAGE_SIZE = 224

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
# 2. Baseline ResNet18
# =========================================================

def build_baseline():

    model = models.resnet18(
        weights=None
    )

    feature_dim = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        feature_dim,
        NUM_CHARACTER_CLASSES
    )

    return model


# =========================================================
# 3. Partial Sharing ResNet18
#
# 最终模型：
#
# conv1
# layer1
# layer2
# layer3
#
# 全部共享
#
# layer4开始分为：
#
# Character layer4
# Source layer4
#
# =========================================================

class PartialSharingLayer3(nn.Module):

    def __init__(
        self,
        num_character_classes,
        num_source_classes
    ):

        super().__init__()


        base = models.resnet18(
            weights=None
        )


        # =================================================
        # Shared layers
        # =================================================

        self.conv1 = (
            base.conv1
        )

        self.bn1 = (
            base.bn1
        )

        self.relu = (
            base.relu
        )

        self.maxpool = (
            base.maxpool
        )

        self.layer1 = (
            base.layer1
        )

        self.layer2 = (
            base.layer2
        )

        self.layer3 = (
            base.layer3
        )


        # =================================================
        # Task-specific layer4
        # =================================================

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


        # =================================================
        # Pooling
        # =================================================

        self.avgpool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # =================================================
        # Character Head
        # =================================================

        self.character_head = (
            nn.Linear(
                512,
                num_character_classes
            )
        )


        # =================================================
        # Source Head
        # =================================================

        self.source_head = (
            nn.Linear(
                512,
                num_source_classes
            )
        )


    def forward(
        self,
        x
    ):

        # =================================================
        # Shared backbone
        # =================================================

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)


        # =================================================
        # Character Branch
        # =================================================

        character_feature = (
            self.character_layer4(
                x
            )
        )

        character_feature = (
            self.avgpool(
                character_feature
            )
        )

        character_feature = (
            torch.flatten(
                character_feature,
                1
            )
        )


        character_output = (
            self.character_head(
                character_feature
            )
        )


        # =================================================
        # Source Branch
        # =================================================

        source_feature = (
            self.source_layer4(
                x
            )
        )

        source_feature = (
            self.avgpool(
                source_feature
            )
        )

        source_feature = (
            torch.flatten(
                source_feature,
                1
            )
        )


        source_output = (
            self.source_head(
                source_feature
            )
        )


        return (
            character_output,
            source_output
        )


# =========================================================
# 4. 参数量函数
# =========================================================

def count_params(
    model
):

    total = sum(

        p.numel()

        for p
        in model.parameters()
    )


    trainable = sum(

        p.numel()

        for p
        in model.parameters()

        if p.requires_grad
    )


    return (
        total,
        trainable
    )


# =========================================================
# 5. 格式化数字
# =========================================================

def format_number(
    number
):

    if number >= 1e9:

        return (
            f"{number / 1e9:.4f} G"
        )

    elif number >= 1e6:

        return (
            f"{number / 1e6:.4f} M"
        )

    elif number >= 1e3:

        return (
            f"{number / 1e3:.4f} K"
        )

    else:

        return (
            f"{number:.0f}"
        )


# =========================================================
# 6. 创建测试输入
# =========================================================

dummy_input = torch.randn(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
).to(device)


# =========================================================
# 7. Baseline
# =========================================================

print()

print(
    "=========================================="
)

print(
    "ResNet18 Baseline"
)

print(
    "=========================================="
)


baseline = build_baseline()

baseline = baseline.to(
    device
)

baseline.eval()


baseline_params, baseline_trainable = (
    count_params(
        baseline
    )
)


baseline_macs, thop_params = profile(

    baseline,

    inputs=(
        dummy_input,
    ),

    verbose=False
)


# 采用常见定义：
#
# 1 MAC ≈ 2 FLOPs

baseline_flops = (
    baseline_macs
    * 2
)


print(
    "Parameters:",
    baseline_params
)

print(
    "Parameters(M):",
    f"{baseline_params / 1e6:.4f}"
)

print(
    "Trainable Parameters:",
    baseline_trainable
)


print(
    "MACs:",
    format_number(
        baseline_macs
    )
)

print(
    "FLOPs:",
    format_number(
        baseline_flops
    )
)


# =========================================================
# 8. Partial Sharing
# =========================================================

print()

print(
    "=========================================="
)

print(
    "Partial Sharing ResNet18"
)

print(
    "=========================================="
)


partial = PartialSharingLayer3(

    NUM_CHARACTER_CLASSES,

    NUM_SOURCE_CLASSES
)


partial = partial.to(
    device
)

partial.eval()


partial_params, partial_trainable = (
    count_params(
        partial
    )
)


partial_macs, thop_params = profile(

    partial,

    inputs=(
        dummy_input,
    ),

    verbose=False
)


partial_flops = (
    partial_macs
    * 2
)


print(
    "Parameters:",
    partial_params
)

print(
    "Parameters(M):",
    f"{partial_params / 1e6:.4f}"
)

print(
    "Trainable Parameters:",
    partial_trainable
)


print(
    "MACs:",
    format_number(
        partial_macs
    )
)

print(
    "FLOPs:",
    format_number(
        partial_flops
    )
)


# =========================================================
# 9. 相对增长
# =========================================================

param_increase = (

    (
        partial_params
        - baseline_params
    )

    / baseline_params

    * 100
)


mac_increase = (

    (
        partial_macs
        - baseline_macs
    )

    / baseline_macs

    * 100
)


flop_increase = (

    (
        partial_flops
        - baseline_flops
    )

    / baseline_flops

    * 100
)


print()

print(
    "=========================================="
)

print(
    "Complexity Comparison"
)

print(
    "=========================================="
)


print(
    f"Parameter Increase: "
    f"{param_increase:+.2f}%"
)


print(
    f"MAC Increase: "
    f"{mac_increase:+.2f}%"
)


print(
    f"FLOPs Increase: "
    f"{flop_increase:+.2f}%"
)


# =========================================================
# 10. 最终表格格式
# =========================================================

print()

print(
    "=========================================="
)

print(
    "Paper Table"
)

print(
    "=========================================="
)


print(
    "Method | Params(M) | MACs(G) | FLOPs(G)"
)


print(

    "ResNet18 | "

    f"{baseline_params / 1e6:.4f} | "

    f"{baseline_macs / 1e9:.4f} | "

    f"{baseline_flops / 1e9:.4f}"
)


print(

    "Partial Sharing | "

    f"{partial_params / 1e6:.4f} | "

    f"{partial_macs / 1e9:.4f} | "

    f"{partial_flops / 1e9:.4f}"
)
