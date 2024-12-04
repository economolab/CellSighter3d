import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


def metric(gt, pred, classes_for_cm, colorbar=True):
    sns.set(font_scale=1.8)  # Moderate font scale for better balance
    cm_normed_recall = confusion_matrix(gt, pred, labels=classes_for_cm, normalize="true") * 100
    cm = confusion_matrix(gt, pred, labels=classes_for_cm)

    num_classes = len(classes_for_cm)
    square_size = max(1, 40 // num_classes)  # Dynamically scale figure size
    plt.figure(figsize=(square_size * num_classes, square_size * num_classes))  # Adjust dynamically

    cmap = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Blues(np.arange(255))])
    annot_labels = cm_normed_recall.round(1).astype(str)
    annot_labels = pd.DataFrame(annot_labels) + "\n (" + pd.DataFrame(cm).astype(str) + ")"

    annot_mask = cm_normed_recall.round(1) <= 0.1
    annot_labels[annot_mask] = ""

    sns.heatmap(
        cm_normed_recall.T,
        annot=annot_labels.T,
        fmt='',
        cbar=colorbar,
        cmap=cmap,
        linewidths=0.8,  # Subtle gridlines
        vmin=0,
        vmax=100,
        linecolor='black',
        square=True
    )

    plt.xticks(rotation=45, fontsize=15)  # Fine-tuned label font size
    plt.yticks(rotation=0, fontsize=15)
    plt.xlabel("Clustering and gating", fontsize=18)
    plt.ylabel("CellSighter", fontsize=18)
    plt.tight_layout()  # Ensure no clipping


# Replace the following line with the path to your CSV file
results = pd.read_csv(r"D:\Cellsighter Data\val_results_700.csv") #Fill in the path to your results file
classes_for_cm = np.unique(np.concatenate([results["label"], results["pred"]]))
metric(results["label"], results["pred"], classes_for_cm)
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()
