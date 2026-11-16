import os
import gc
import json
import copy

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
# 1. 基本配置
# =====================================================

ROOT = "/root/calligraphy_sci"

TEST_CSV = os.path.join(
    ROOT, "dual_splits", "dual_test.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "results", "ablation_2x2_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 2026]

NUM_CHAR = 6249
NUM_SOURCE = 19

BATCH_SIZE = 128
NUM_WORKERS = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# 2. 实验列表
#
# fixed partial 已完成最终 Test，
# 此次只评估另外三个训练配置
# =====================================================

EXPERIMENTS = [
    {
        "name": "equal_fullshare",
        "architecture": "full",
        "source_weight": 1.0,
        "dir_prefix": "equal_fullshare_seed"
    },
    {
        "name": "equal_partial",
        "architecture": "partial",
        "source_weight": 1.0,
        "dir_prefix": "equal_partial_layer3_seed"
    },
    {
        "name": "fixed_fullshare",
        "architecture": "full",
        "source_weight": 0.1,
        "dir_prefix": "fixed_fullshare_aux010_seed"
    }
]


# =====================================================
# 3. Test Dataset
# =====================================================

class DualTestDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path).reset_index(
            drop=True
        )

        required = [
            "path",
            "character_label",
            "source_label"
        ]

        for col in required:
            if col not in self.df.columns:
                raise ValueError(
                    f"Test CSV 缺少列: {col}"
                )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        with Image.open(str(row["path"])) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return (
            image,
            int(row["character_label"]),
            int(row["source_label"]),
            idx
        )


test_dataset = DualTestDataset(TEST_CSV)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

assert len(test_dataset) == 13804, (
    "Test 数量与此前实验不一致，请检查划分文件"
)


# =====================================================
# 4. Full Sharing 网络
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
# 5. Partial Sharing 网络
#
# 与前面等权部分共享脚本的结构及命名一致
# =====================================================

class PartialSharingResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.resnet18(
            weights=None
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        char_feature = self.character_layer4(x)
        char_feature = self.avgpool(char_feature)
        char_feature = torch.flatten(
            char_feature, 1
        )

        source_feature = self.source_layer4(x)
        source_feature = self.avgpool(source_feature)
        source_feature = torch.flatten(
            source_feature, 1
        )

        char_output = self.character_head(
            char_feature
        )

        source_output = self.source_head(
            source_feature
        )

        return char_output, source_output


# =====================================================
# 6. 创建并严格加载最佳模型
# =====================================================

def load_model(exp, seed):

    experiment_dir = (
        exp["dir_prefix"] + str(seed)
    )

    checkpoint_path = os.path.join(
        ROOT,
        "checkpoints",
        experiment_dir,
        "best_model.pth"
    )

    summary_path = os.path.join(
        ROOT,
        "results",
        experiment_dir,
        "best_summary.json"
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {checkpoint_path}"
        )

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"找不到验证结果: {summary_path}"
        )

    with open(
        summary_path, "r", encoding="utf-8"
    ) as f:
        val_summary = json.load(f)

    if int(val_summary["seed"]) != seed:
        raise ValueError(
            "Checkpoint 对应的 seed 不正确"
        )

    weight = float(
        val_summary["source_weight"]
    )

    if abs(weight - exp["source_weight"]) > 1e-8:
        raise ValueError(
            "Source weight 与实验配置不一致"
        )

    if exp["architecture"] == "full":
        model = FullSharingResNet18()
    else:
        model = PartialSharingResNet18()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # 严格匹配：避免意外遗漏模型参数
    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Val Best Epoch:", val_summary["best_epoch"])

    return model, checkpoint_path


# =====================================================
# 7. Test 评估
# =====================================================

@torch.inference_mode()
def evaluate(model, experiment_name, seed):

    model.eval()

    all_char_true = []
    all_char_pred = []

    all_source_true = []
    all_source_pred = []

    all_indices = []

    char_top5_correct = 0
    source_top5_correct = 0

    for images, y_char, y_source, indices in tqdm(
        test_loader,
        desc=f"{experiment_name} seed={seed}"
    ):

        images = images.to(
            device, non_blocking=True
        )

        y_char = y_char.to(
            device, non_blocking=True
        )

        y_source = y_source.to(
            device, non_blocking=True
        )

        char_logits, source_logits = model(images)

        pred_char = char_logits.argmax(dim=1)
        pred_source = source_logits.argmax(dim=1)

        char_top5 = char_logits.topk(
            5, dim=1
        ).indices

        source_top5 = source_logits.topk(
            5, dim=1
        ).indices

        char_top5_correct += (
            char_top5 == y_char.unsqueeze(1)
        ).any(dim=1).sum().item()

        source_top5_correct += (
            source_top5 == y_source.unsqueeze(1)
        ).any(dim=1).sum().item()

        all_char_true.extend(
            y_char.cpu().tolist()
        )

        all_char_pred.extend(
            pred_char.cpu().tolist()
        )

        all_source_true.extend(
            y_source.cpu().tolist()
        )

        all_source_pred.extend(
            pred_source.cpu().tolist()
        )

        all_indices.extend(
            indices.tolist()
        )

    total = len(all_char_true)

    char_top1 = (
        100.0
        * np.mean(
            np.array(all_char_true)
            == np.array(all_char_pred)
        )
    )

    source_top1 = (
        100.0
        * np.mean(
            np.array(all_source_true)
            == np.array(all_source_pred)
        )
    )

    char_top5 = (
        100.0 * char_top5_correct / total
    )

    source_top5 = (
        100.0 * source_top5_correct / total
    )

    char_f1 = f1_score(
        all_char_true,
        all_char_pred,
        average="macro",
        labels=list(range(NUM_CHAR)),
        zero_division=0
    )

    source_f1 = f1_score(
        all_source_true,
        all_source_pred,
        average="macro",
        labels=list(range(NUM_SOURCE)),
        zero_division=0
    )

    # 保存逐样本预测，以便后续错误分析
    predictions = pd.DataFrame({
        "index": all_indices,
        "character_true": all_char_true,
        "character_pred": all_char_pred,
        "source_true": all_source_true,
        "source_pred": all_source_pred
    })

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{experiment_name}_seed{seed}_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "experiment": experiment_name,
        "seed": seed,
        "num_samples": total,
        "char_top1": float(char_top1),
        "char_top5": float(char_top5),
        "char_macro_f1": float(char_f1),
        "source_top1": float(source_top1),
        "source_top5": float(source_top5),
        "source_macro_f1": float(source_f1)
    }


# =====================================================
# 8. 执行全部 Test
# =====================================================

print()
print("=" * 55)
print("2x2 ABLATION TEST")
print("=" * 55)

print("Device:", device)
print("Test Samples:", len(test_dataset))

results = []

for exp in EXPERIMENTS:

    for seed in SEEDS:

        print()
        print("=" * 55)
        print("Experiment:", exp["name"])
        print("Seed:", seed)
        print("=" * 55)

        model, checkpoint_path = load_model(
            exp, seed
        )

        result = evaluate(
            model,
            exp["name"],
            seed
        )

        result["checkpoint"] = checkpoint_path

        results.append(result)

        print()
        print(
            f"Character Top-1: "
            f"{result['char_top1']:.4f}%"
        )

        print(
            f"Character Macro-F1: "
            f"{result['char_macro_f1']:.6f}"
        )

        print(
            f"Source Top-1: "
            f"{result['source_top1']:.4f}%"
        )

        print(
            f"Source Macro-F1: "
            f"{result['source_macro_f1']:.6f}"
        )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =====================================================
# 9. 保存逐 seed 结果
# =====================================================

result_df = pd.DataFrame(results)

result_path = os.path.join(
    OUTPUT_DIR,
    "test_results_3seeds.csv"
)

result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 10. Mean ± Std
# =====================================================

metrics = [
    "char_top1",
    "char_top5",
    "char_macro_f1",
    "source_top1",
    "source_top5",
    "source_macro_f1"
]

summary_rows = []

for exp in EXPERIMENTS:

    name = exp["name"]

    sub = result_df[
        result_df["experiment"] == name
    ]

    row = {
        "experiment": name,
        "source_weight": exp["source_weight"],
        "data_source": "new_test"
    }

    for metric in metrics:

        values = sub[metric].to_numpy(
            dtype=float
        )

        row[f"{metric}_mean"] = float(
            np.mean(values)
        )

        row[f"{metric}_std"] = float(
            np.std(values, ddof=1)
        )

    summary_rows.append(row)


# =====================================================
# 11. 合并此前已经完成的 Ours Test
#
# 这些是此前冻结模型后的 Test Mean ± Std。
# 为避免原始 Partial Sharing 与新脚本模型命名不同，
# 这里复用已经报告的测试结果，不重新加载 checkpoint。
# =====================================================

ours_reference = {
    "experiment": "fixed_partial_ours",
    "source_weight": 0.1,
    "data_source": "previous_final_test",
    "char_top1_mean": 96.5662,
    "char_top1_std": 0.1206,
    "char_top5_mean": 99.2200,
    "char_top5_std": 0.0399,
    "char_macro_f1_mean": 0.962637,
    "char_macro_f1_std": 0.001417,
    "source_top1_mean": 96.7256,
    "source_top1_std": 0.0735,
    "source_top5_mean": float("nan"),
    "source_top5_std": float("nan"),
    "source_macro_f1_mean": 0.970277,
    "source_macro_f1_std": 0.000637
}

summary_rows.append(ours_reference)

summary_df = pd.DataFrame(summary_rows)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "test_summary_2x2.csv"
)

summary_json = os.path.join(
    OUTPUT_DIR,
    "test_summary_2x2.json"
)

summary_df.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary_rows,
        f,
        ensure_ascii=False,
        indent=4
    )


# =====================================================
# 12. 输出最终 2x2 对比
# =====================================================

print()
print("=" * 65)
print("FINAL 2x2 TEST SUMMARY")
print("=" * 65)

for row in summary_rows:

    print()
    print("Method:", row["experiment"])

    print(
        "Character Top-1:",
        f"{row['char_top1_mean']:.4f}"
        f" ± {row['char_top1_std']:.4f}%"
    )

    print(
        "Source Top-1:",
        f"{row['source_top1_mean']:.4f}"
        f" ± {row['source_top1_std']:.4f}%"
    )

    print(
        "Character Macro-F1:",
        f"{row['char_macro_f1_mean']:.6f}"
        f" ± {row['char_macro_f1_std']:.6f}"
    )

    print(
        "Source Macro-F1:",
        f"{row['source_macro_f1_mean']:.6f}"
        f" ± {row['source_macro_f1_std']:.6f}"
    )


# =====================================================
# 13. 固定权重下的结构收益
# =====================================================

summary_by_name = {
    row["experiment"]: row
    for row in summary_rows
}

print()
print("=" * 65)
print("SHARING STRATEGY IMPROVEMENT")
print("=" * 65)

for weight, full_name, partial_name in [
    (
        1.0,
        "equal_fullshare",
        "equal_partial"
    ),
    (
        0.1,
        "fixed_fullshare",
        "fixed_partial_ours"
    )
]:

    full = summary_by_name[full_name]
    partial = summary_by_name[partial_name]

    char_gain = (
        partial["char_top1_mean"]
        - full["char_top1_mean"]
    )

    source_gain = (
        partial["source_top1_mean"]
        - full["source_top1_mean"]
    )

    print()
    print(f"Lambda = {weight}")
    print(f"Character Improvement: {char_gain:+.4f} pp")
    print(f"Source Improvement: {source_gain:+.4f} pp")


print()
print("Detailed results:", result_path)
print("Final CSV:", summary_csv)
print("Final JSON:", summary_json)

print()
print("2x2 ABLATION TEST COMPLETE")
