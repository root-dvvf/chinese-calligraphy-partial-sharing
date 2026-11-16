import os
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score
from tqdm import tqdm


# =========================================================
# 1. 配置
# =========================================================

NUM_CLASSES = 6249
BATCH_SIZE = 128
NUM_WORKERS = 8
IMAGE_SIZE = 224

TEST_CSV = "/root/calligraphy_sci/splits/test.csv"

MODEL_PATH = (
    "/root/calligraphy_sci/checkpoints/"
    "resnet18_cb_loss/best_model.pth"
)

RESULT_DIR = (
    "/root/calligraphy_sci/results/"
    "resnet18_cb_loss"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# =========================================================
# 2. Dataset
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

        return (
            image,
            label,
            img_path
        )


# =========================================================
# 3. 测试预处理
# =========================================================

test_transform = transforms.Compose([

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
# 4. DataLoader
# =========================================================

test_dataset = CalligraphyDataset(
    TEST_CSV,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
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


# =========================================================
# 6. 模型
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
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
# 7. Test
# =========================================================

correct_top1 = 0
correct_top5 = 0
total = 0

all_labels = []
all_preds = []
all_paths = []


with torch.no_grad():

    for (
        images,
        labels,
        paths

    ) in tqdm(
        test_loader,
        desc="Test"
    ):

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


        all_labels.extend(
            labels
            .cpu()
            .numpy()
        )

        all_preds.extend(
            pred_top1
            .cpu()
            .numpy()
        )

        all_paths.extend(
            paths
        )


# =========================================================
# 8. 指标
# =========================================================

top1 = (
    100.0
    * correct_top1
    / total
)

top5 = (
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


print()

print(
    "========== Test Result =========="
)

print(
    "Test Samples:",
    total
)

print(
    f"Test Top-1: "
    f"{top1:.2f}%"
)

print(
    f"Test Top-5: "
    f"{top5:.2f}%"
)

print(
    f"Test Macro-F1: "
    f"{macro_f1:.4f}"
)


# =========================================================
# 9. 保存总体结果
# =========================================================

results = {

    "model":
        "ResNet18 + Class-Balanced Loss",

    "test_samples":
        total,

    "test_top1":
        top1,

    "test_top5":
        top5,

    "test_macro_f1":
        macro_f1
}


result_path = os.path.join(
    RESULT_DIR,
    "test_results.json"
)


with open(
    result_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# 10. 保存每张图片预测
# =========================================================

prediction_df = pd.DataFrame({

    "path":
        all_paths,

    "true_label":
        all_labels,

    "pred_label":
        all_preds
})


prediction_path = os.path.join(
    RESULT_DIR,
    "test_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


print()

print(
    "结果已保存:"
)

print(
    result_path
)

print(
    prediction_path
)
