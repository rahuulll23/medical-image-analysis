import torch, os, json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def save_checkpoint(state, filename="saved_models/checkpoint.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def plot_confusion_matrix(cm, labels, out_path="saved_models/confusion_matrix.png"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
