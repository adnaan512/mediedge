"""
MediEdge — Generate Report Figures.

Creates:
  1. Accuracy vs. Model Size tradeoff curve
  2. CPU Latency comparison bar chart
  3. Confusion Matrix for ResNet-18 and MobileNetV2 (Distilled)
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.makedirs("results/figures", exist_ok=True)
sns.set_theme(style="whitegrid", font="sans-serif")

# Load results
models_to_plot = [
    ("ResNet-18 (Baseline)", "results/resnet18_dermamnist_results.json", "#1f77b4", "s"),
    ("EfficientNet-Lite", "results/efficientnet_lite_dermamnist_results.json", "#ff7f0e", "^"),
    ("Small CNN", "results/cnn_dermamnist_results.json", "#2ca02c", "o"),
    ("MobileNetV2 (Base)", "results/mobilenetv2_dermamnist_results.json", "#d62728", "D"),
    ("MobileNetV2 + Distill", "results/mobilenetv2_dermamnist_distilled_results.json", "#9467bd", "P"),
    ("MobileNetV2 + INT8", "results/mobilenetv2_dermamnist_int8_results.json", "#8c564b", "X"),
    ("MobileNetV2 + Prune", "results/mobilenetv2_dermamnist_pruned_results.json", "#e377c2", "v"),
    ("MobileNetV2 Combined", "results/mobilenetv2_dermamnist_combined_results.json", "#17becf", "*"),
]

data = []
for label, path, color, marker in models_to_plot:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
            data.append({
                "label": label,
                "accuracy": d["accuracy"] * 100,
                "auc": d["auc"],
                "size_mb": d["size_mb"],
                "latency": d["latency_mean_ms"],
                "color": color,
                "marker": marker,
                "cm": d.get("confusion_matrix")
            })

# ── Figure 1: Accuracy vs Model Size Tradeoff ─────────────────────────────────
plt.figure(figsize=(9, 6), dpi=300)
for item in data:
    plt.scatter(
        item["size_mb"], item["accuracy"],
        color=item["color"], marker=item["marker"], s=180, edgecolors="black", linewidth=1.2, zorder=4
    )
    # Smart offsets for labels
    offset_x = 0.5
    offset_y = 0.2
    if "ResNet" in item["label"]:
        offset_x = -11.0
    elif "EfficientNet" in item["label"]:
        offset_x = -7.5
    elif "Small" in item["label"]:
        offset_x = 0.8
    elif "Combined" in item["label"]:
        offset_y = -0.55
        offset_x = 0.5
    elif "INT8" in item["label"]:
        offset_y = 0.35
        offset_x = 0.5

    plt.annotate(
        f"{item['label']} ({item['accuracy']:.1f}%, {item['size_mb']:.1f}MB)",
        (item["size_mb"] + offset_x, item["accuracy"] + offset_y),
        fontsize=9, weight="bold"
    )

plt.axvline(x=5.0, color="gray", linestyle="--", alpha=0.5, label="Edge Budget (<5 MB)")
plt.title("MediEdge: Accuracy vs. Model Size Trade-off on DermaMNIST", fontsize=14, weight="bold", pad=12)
plt.xlabel("Model Size (MB) [Lower is Better]", fontsize=12, labelpad=8)
plt.ylabel("Test Accuracy (%) [Higher is Better]", fontsize=12, labelpad=8)
plt.ylim(70, 80)
plt.xlim(-1, 46)
plt.tight_layout()
plt.savefig("results/figures/fig1_accuracy_vs_size.png")
plt.close()
print("Saved results/figures/fig1_accuracy_vs_size.png")

# ── Figure 2: CPU Latency Comparison Bar Chart ────────────────────────────────
plt.figure(figsize=(10, 5), dpi=300)
labels = [d["label"] for d in data]
latencies = [d["latency"] for d in data]
colors = [d["color"] for d in data]

bars = plt.barh(labels, latencies, color=colors, edgecolor="black", height=0.6)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f} ms", va="center", ha="left", fontsize=9, weight="bold")

plt.xlabel("CPU Latency (ms per image) [Lower is Better]", fontsize=12, labelpad=8)
plt.title("MediEdge: CPU Inference Latency per Image", fontsize=14, weight="bold", pad=12)
plt.xlim(0, max(latencies) + 6)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/figures/fig2_latency_comparison.png")
plt.close()
print("Saved results/figures/fig2_latency_comparison.png")

# ── Figure 3: Confusion Matrix for ResNet-18 vs Distilled MobileNetV2 ─────────
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for idx, (target_name, ax) in enumerate([("ResNet-18 (Baseline)", axes[0]), ("MobileNetV2 + Distill", axes[1])]):
    match = next(d for d in data if d["label"] == target_name)
    cm = np.array(match["cm"])
    # Normalized by row (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_title(f"{target_name}\nAcc: {match['accuracy']:.2f}% | AUC: {match['auc']:.4f}", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)

plt.suptitle("MediEdge: Confusion Matrix Analysis (DermaMNIST)", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/fig3_confusion_matrix.png")
plt.close()
print("Saved results/figures/fig3_confusion_matrix.png")
