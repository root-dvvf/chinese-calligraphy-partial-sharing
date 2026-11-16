import os
import json
import time
import random
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


# ==============================
# 1. Config
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

ROOT = "/root/calligraphy_sci"

SEED = args.seed
SOURCE_WEIGHT = 0.10

NUM_CHAR = 6249
NUM_SOURCE = 19

EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WORKERS = 8

NAME = f"fixed_fullshare_aux010_seed{SEED}"

RESULT_DIR = os.path.join(ROOT, "results", NAME)
CKPT_DIR = os.path.join(ROOT, "checkpoints", NAME)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(
    ROOT, "dual_splits", "dual_train.csv"
)
VAL_CSV = os.path.join(
    ROOT, "dual_splits", "dual_val.csv"
)


# ==============================
# 2. Random seed
# ==============================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==============================
# 3. Dataset
# ==============================

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


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_dataset = DualDataset(TRAIN_CSV, train_transform)
val_dataset = DualDataset(VAL_CSV, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True
)


# ==============================
# 4. Full Sharing ResNet18
# ==============================

class FullSharingResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.character_head = nn.Linear(512, NUM_CHAR)
        self.source_head = nn.Linear(512, NUM_SOURCE)

    def forward(self, x):

        feature = self.backbone(x)

        char_logits = self.character_head(feature)
        source_logits = self.source_head(feature)

        return char_logits, source_logits


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FullSharingResNet18().to(device)

params = sum(p.numel() for p in model.parameters())

char_loss_fn = nn.CrossEntropyLoss()
source_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ==============================
# 5. Save configuration
# ==============================

config = {
    "model": "Full Sharing ResNet18",
    "seed": SEED,
    "source_weight": SOURCE_WEIGHT,
    "loss": "L_char + 0.1 * L_source",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "input_size": 224,
    "pretrained": False,
    "augmentation": "RandomRotation(5)",
    "normalization": "ImageNet mean/std",
    "model_selection": "Val Character Top-1",
    "train_csv": TRAIN_CSV,
    "val_csv": VAL_CSV,
    "params": params
}

with open(
    os.path.join(RESULT_DIR, "config.json"),
    "w", encoding="utf-8"
) as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


# ==============================
# 6. Training
# ==============================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    char_correct = 0
    source_correct = 0
    total = 0

    for images, y_char, y_source in tqdm(
        train_loader, desc="Train"
    ):

        images = images.to(device, non_blocking=True)
        y_char = y_char.to(device, non_blocking=True)
        y_source = y_source.to(device, non_blocking=True)

        optimizer.zero_grad()

        out_char, out_source = model(images)

        loss_char = char_loss_fn(out_char, y_char)
        loss_source = source_loss_fn(out_source, y_source)

        loss = loss_char + SOURCE_WEIGHT * loss_source

        loss.backward()
        optimizer.step()

        n = images.size(0)
        total += n
        running_loss += loss.item() * n

        char_correct += (
            out_char.argmax(1) == y_char
        ).sum().item()

        source_correct += (
            out_source.argmax(1) == y_source
        ).sum().item()

    return {
        "train_loss": running_loss / total,
        "train_char_top1": char_correct / total * 100,
        "train_source_top1": source_correct / total * 100
    }


# ==============================
# 7. Validation
# ==============================

@torch.no_grad()
def validate():

    model.eval()

    total = 0
    char_correct = 0
    char_top5_correct = 0
    source_correct = 0

    char_true = []
    char_pred = []

    source_true = []
    source_pred = []

    for images, y_char, y_source in tqdm(
        val_loader, desc="Val"
    ):

        images = images.to(device, non_blocking=True)
        y_char = y_char.to(device, non_blocking=True)
        y_source = y_source.to(device, non_blocking=True)

        out_char, out_source = model(images)

        pred_char = out_char.argmax(1)
        pred_source = out_source.argmax(1)

        n = images.size(0)
        total += n

        char_correct += (
            pred_char == y_char
        ).sum().item()

        source_correct += (
            pred_source == y_source
        ).sum().item()

        top5 = out_char.topk(5, dim=1).indices
        char_top5_correct += (
            top5 == y_char.unsqueeze(1)
        ).any(dim=1).sum().item()

        char_true.extend(y_char.cpu().tolist())
        char_pred.extend(pred_char.cpu().tolist())

        source_true.extend(y_source.cpu().tolist())
        source_pred.extend(pred_source.cpu().tolist())

    return {
        "val_char_top1": char_correct / total * 100,
        "val_char_top5": char_top5_correct / total * 100,
        "val_char_macro_f1": f1_score(
            char_true, char_pred,
            average="macro", zero_division=0
        ),
        "val_source_top1": source_correct / total * 100,
        "val_source_macro_f1": f1_score(
            source_true, source_pred,
            average="macro", zero_division=0
        )
    }


# ==============================
# 8. Main training loop
# ==============================

print("=" * 55)
print("Experiment:", NAME)
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Device:", device)
print("Params(M):", round(params / 1e6, 4))
print("Loss: L_char + 0.1 * L_source")
print("=" * 55)

best_char_top1 = -1.0
best_summary = None
history = []

for epoch in range(1, EPOCHS + 1):

    start = time.time()

    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

    train_result = train_one_epoch()
    val_result = validate()

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    log = {
        "epoch": epoch,
        **train_result,
        **val_result,
        "learning_rate": current_lr,
        "epoch_time": time.time() - start
    }

    history.append(log)

    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, "train_log.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Train Char: {train_result['train_char_top1']:.4f}%")
    print(f"Train Source: {train_result['train_source_top1']:.4f}%")

    print(f"Val Char Top-1: {val_result['val_char_top1']:.6f}%")
    print(f"Val Char Top-5: {val_result['val_char_top5']:.6f}%")
    print(f"Val Char F1: {val_result['val_char_macro_f1']:.6f}")
    print(f"Val Source Top-1: {val_result['val_source_top1']:.6f}%")
    print(f"Val Source F1: {val_result['val_source_macro_f1']:.6f}")

    # 只按 Character 验证集准确率选择最佳模型
    if val_result["val_char_top1"] > best_char_top1:

        best_char_top1 = val_result["val_char_top1"]

        best_summary = {
            "seed": SEED,
            "best_epoch": epoch,
            "source_weight": SOURCE_WEIGHT,
            "params": params,
            **val_result
        }

        torch.save(
            model.state_dict(),
            os.path.join(CKPT_DIR, "best_model.pth")
        )

        with open(
            os.path.join(RESULT_DIR, "best_summary.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                best_summary, f,
                indent=4, ensure_ascii=False
            )

        print("保存新的最佳 Full Sharing 模型")


print("\n" + "=" * 55)
print("FIXED FULL SHARING TRAINING COMPLETE")
print("=" * 55)

print("Seed:", SEED)
print("Best Epoch:", best_summary["best_epoch"])
print("Val Character Top-1:", best_summary["val_char_top1"])
print("Val Character Macro-F1:", best_summary["val_char_macro_f1"])
print("Val Source Top-1:", best_summary["val_source_top1"])
print("Val Source Macro-F1:", best_summary["val_source_macro_f1"])

print("Result Directory:", RESULT_DIR)
print("Checkpoint Directory:", CKPT_DIR)
