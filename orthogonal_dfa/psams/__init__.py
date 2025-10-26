import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc(prediction, target, submask=None, submask_name=""):
    """
    Plots the ROC curve for the PSAMs output against the target.

    :param prediction: torch.Tensor of shape (batch_size, num_psams) representing log probabilities
    :param target: binary torch.Tensor of shape (batch_size,)
    """
    if submask is not None:
        prediction = prediction[submask]
        target = target[submask]
    topk_thresh = np.quantile(prediction, 1 - target.mean())
    topk_acc = ((prediction >= topk_thresh) == target).mean()
    fpr, tpr, _ = roc_curve(target, prediction)
    auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC Curve: AUC = {auc:.3f}; Top-k Acc = {topk_acc:.2%}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {submask_name}")
    plt.legend()
    plt.show()
