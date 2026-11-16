import copy
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# =========================================================
# 1. 配置
# =========================================================

NUM_CHARACTER_CLASSES = 6249
NUM_SOURCE_CLASSES = 19

IMAGE_SIZE = 224

BATCH_SIZES = [1, 128]

WARMUP_ITERS = 100
TEST_ITERS = 500

BASELINE_CKPT = (
    "/root/calligraphy_sci/"
    "checkpoints/baseline_resnet18_seed42/"
    "best_model.pth"
)

OURS_CKPT = (
    "/root/calligraphy_sci/"
    "checkpoints/partial_layer3_aux010_seed42/"
    "best_model.pth"
)

RESULT_PATH = (
    "/root/calligraphy_sci/"
    "results/inference_speed_results.json"
)


# =========================================================
# 2. Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    torch.backends.cudnn.benchmark = True


# =========================================================
# 3. Baseline
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
# 4. Partial Sharing
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


        # Shared backbone

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3


        # Task-specific layer4

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


        self.avgpool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.character_head = nn.Linear(
            512,
            num_character_classes
        )

        self.source_head = nn.Linear(
            512,
            num_source_classes
        )


    def forward(
        self,
        x
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        # Character branch

        char_x = self.character_layer4(
            x
        )

        char_x = self.avgpool(
            char_x
        )

        char_x = torch.flatten(
            char_x,
            1
        )

        character_output = (
            self.character_head(
                char_x
            )
        )


        # Source branch

        source_x = self.source_layer4(
            x
        )

        source_x = self.avgpool(
            source_x
        )

        source_x = torch.flatten(
            source_x,
            1
        )

        source_output = (
            self.source_head(
                source_x
            )
        )


        return (
            character_output,
            source_output
        )


# =========================================================
# 5. 加载模型
# =========================================================

baseline = build_baseline().to(
    device
)

baseline.load_state_dict(

    torch.load(
        BASELINE_CKPT,
        map_location=device,
        weights_only=True
    )
)

baseline.eval()


ours = PartialSharingLayer3(
    NUM_CHARACTER_CLASSES,
    NUM_SOURCE_CLASSES
).to(device)

ours.load_state_dict(

    torch.load(
        OURS_CKPT,
        map_location=device,
        weights_only=True
    )
)

ours.eval()


# =========================================================
# 6. 推理测速
# =========================================================

def benchmark_model(
    model,
    model_name,
    batch_size
):

    print()
    print(
        "========================================"
    )

    print(
        f"{model_name} | Batch Size = {batch_size}"
    )

    print(
        "========================================"
    )


    dummy_input = torch.randn(

        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,

        device=device
    )


    # =====================================================
    # Warm-up
    # =====================================================

    with torch.inference_mode():

        for _ in range(
            WARMUP_ITERS
        ):

            _ = model(
                dummy_input
            )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    # =====================================================
    # 正式测速
    # =====================================================

    times_ms = []


    with torch.inference_mode():

        for _ in range(
            TEST_ITERS
        ):


            if torch.cuda.is_available():

                start_event = (
                    torch.cuda.Event(
                        enable_timing=True
                    )
                )

                end_event = (
                    torch.cuda.Event(
                        enable_timing=True
                    )
                )


                start_event.record()

                _ = model(
                    dummy_input
                )

                end_event.record()

                torch.cuda.synchronize()


                elapsed_ms = (
                    start_event.elapsed_time(
                        end_event
                    )
                )


            else:

                start = (
                    time.perf_counter()
                )


                _ = model(
                    dummy_input
                )


                end = (
                    time.perf_counter()
                )


                elapsed_ms = (
                    (end - start)
                    * 1000
                )


            times_ms.append(
                elapsed_ms
            )


    times_ms = np.array(
        times_ms
    )


    mean_batch_ms = float(
        np.mean(
            times_ms
        )
    )

    std_batch_ms = float(
        np.std(
            times_ms,
            ddof=1
        )
    )


    median_batch_ms = float(
        np.median(
            times_ms
        )
    )


    per_image_ms = (
        mean_batch_ms
        / batch_size
    )


    fps = (
        batch_size
        * 1000.0
        / mean_batch_ms
    )


    print(
        f"Mean Batch Latency: "
        f"{mean_batch_ms:.4f} ms"
    )

    print(
        f"Std Batch Latency: "
        f"{std_batch_ms:.4f} ms"
    )

    print(
        f"Median Batch Latency: "
        f"{median_batch_ms:.4f} ms"
    )

    print(
        f"Per-image Latency: "
        f"{per_image_ms:.4f} ms"
    )

    print(
        f"Throughput / FPS: "
        f"{fps:.2f} images/s"
    )


    return {

        "model":
            model_name,

        "batch_size":
            batch_size,

        "mean_batch_latency_ms":
            mean_batch_ms,

        "std_batch_latency_ms":
            std_batch_ms,

        "median_batch_latency_ms":
            median_batch_ms,

        "per_image_latency_ms":
            per_image_ms,

        "throughput_images_per_second":
            fps
    }


# =========================================================
# 7. 正式测试
# =========================================================

results = []


for batch_size in BATCH_SIZES:

    result = benchmark_model(

        baseline,

        "ResNet18 Baseline",

        batch_size
    )

    results.append(
        result
    )


for batch_size in BATCH_SIZES:

    result = benchmark_model(

        ours,

        "Partial Sharing ResNet18",

        batch_size
    )

    results.append(
        result
    )


# =========================================================
# 8. 保存JSON
# =========================================================

with open(

    RESULT_PATH,

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
# 9. 自动计算相对变化
# =========================================================

def get_result(
    model_name,
    batch_size
):

    for item in results:

        if (
            item["model"]
            == model_name

            and

            item["batch_size"]
            == batch_size
        ):

            return item


print()
print(
    "========================================"
)

print(
    "FINAL SPEED COMPARISON"
)

print(
    "========================================"
)


for batch_size in BATCH_SIZES:


    baseline_result = get_result(

        "ResNet18 Baseline",

        batch_size
    )


    ours_result = get_result(

        "Partial Sharing ResNet18",

        batch_size
    )


    latency_increase = (

        (
            ours_result[
                "mean_batch_latency_ms"
            ]

            -

            baseline_result[
                "mean_batch_latency_ms"
            ]
        )

        /

        baseline_result[
            "mean_batch_latency_ms"
        ]

        * 100
    )


    fps_change = (

        (
            ours_result[
                "throughput_images_per_second"
            ]

            -

            baseline_result[
                "throughput_images_per_second"
            ]
        )

        /

        baseline_result[
            "throughput_images_per_second"
        ]

        * 100
    )


    print()

    print(
        f"Batch Size = {batch_size}"
    )


    print(
        "Baseline Latency:"
    )

    print(
        f"{baseline_result['mean_batch_latency_ms']:.4f} ms"
    )


    print(
        "Ours Latency:"
    )

    print(
        f"{ours_result['mean_batch_latency_ms']:.4f} ms"
    )


    print(
        f"Latency Increase: "
        f"{latency_increase:+.2f}%"
    )


    print(
        "Baseline FPS:"
    )

    print(
        f"{baseline_result['throughput_images_per_second']:.2f}"
    )


    print(
        "Ours FPS:"
    )

    print(
        f"{ours_result['throughput_images_per_second']:.2f}"
    )


    print(
        f"FPS Change: "
        f"{fps_change:+.2f}%"
    )


print()
print(
    "Saved to:",
    RESULT_PATH
)
