import os
import csv
import statistics
from collections import Counter

import matplotlib.pyplot as plt

DATA_DIR = "/root/calligraphy_sci/data/chinese-calligraphy-dataset/chinese-calligraphy-dataset"
OUT_DIR = "/root/calligraphy_sci/results"

os.makedirs(OUT_DIR, exist_ok=True)

class_counts = {}

for name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, name)
    if not os.path.isdir(class_dir):
        continue

    count = 0
    for f in os.listdir(class_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            count += 1

    class_counts[name] = count

counts = list(class_counts.values())

print("类别数:", len(class_counts))
print("总图片数:", sum(counts))
print("最少样本数:", min(counts))
print("最多样本数:", max(counts))
print("平均样本数:", sum(counts) / len(counts))
print("中位数:", statistics.median(counts))

sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])

print("\n样本最少的20类:")
for name, count in sorted_counts[:20]:
    print(name, count)

print("\n样本最多的20类:")
for name, count in sorted_counts[-20:][::-1]:
    print(name, count)

csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "count"])

    for name, count in sorted_counts:
        writer.writerow([name, count])

ranked_counts = sorted(counts, reverse=True)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(ranked_counts) + 1), ranked_counts)
plt.xlabel("Class Rank")
plt.ylabel("Number of Samples")
plt.title("Class Distribution of Chinese Calligraphy Dataset")
plt.tight_layout()

fig_path = os.path.join(OUT_DIR, "class_distribution.png")
plt.savefig(fig_path, dpi=300)
plt.close()

freq_counter = Counter(counts)

freq_csv_path = os.path.join(OUT_DIR, "sample_count_frequency.csv")
with open(freq_csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_count", "num_classes"])

    for sample_count in sorted(freq_counter.keys()):
        writer.writerow([sample_count, freq_counter[sample_count]])

print("\n输出文件:")
print(csv_path)
print(fig_path)
print(freq_csv_path)
