import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

#################################################################################################


def pfbeta_np(gts, preds, beta=1):
    preds = preds.clip(0, 1.0)
    y_true_count = gts.sum()
    ctp = preds[gts == 1].sum()
    cfp = preds[gts == 0].sum()
    beta_squared = beta * beta
    if ctp + cfp == 0:
        c_precision = 0.0
    else:
        c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        ret = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return ret
    else:
        return 0.0


def compute_pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-8)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def _compute_fbeta(precision, recall, beta=1.0):
    if ((beta**2) * precision + recall) == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)


def compute_usual_metrics(gts, preds, beta=1.0, sample_weights=None):
    """Binary prediction only."""
    cfm = metrics.confusion_matrix(
        gts, preds, labels=[0, 1], sample_weight=sample_weights
    )

    tn, fp, fn, tp = cfm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    fbeta = _compute_fbeta(precision, recall, beta=beta)
    f1_score = metrics.f1_score(gts, preds, average="weighted")
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1score": f1_score,
        "fbeta": fbeta,
    }


#################################################################################################


def plot_pr_curve(result, plot_save_path, df):
    _, axs = plt.subplots(1, 2, figsize=(20, 10))

    ############################################################################
    # PRECISION-RECALL CURVE
    ax = axs[0]
    f_scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    ax.plot([0, 1], [0, 1], color="gray", alpha=0.2)

    # overall
    precision, recall, threshold = metrics.precision_recall_curve(df.targets, df.preds)
    auc = metrics.auc(recall, precision)
    ax.plot(recall, precision)
    s = ax.scatter(recall[:-1], precision[:-1], c=threshold, cmap="hsv")
    recall_max, precision_max, f1score_max, threshold_max = (
        result["single_best_recall"],
        result["single_best_precision"],
        result["single_best_f1score"],
        result["single_best_thres"],
    )
    ax.scatter(recall_max, precision_max, s=30, c="k")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    text = ""
    text += f"MAX f1score {f1score_max: 0.5f} @ th = {threshold_max: 0.5f}\n"
    text += (
        f"prec {precision_max: 0.5f}, recall {recall_max: 0.5f}, pr-auc {auc: 0.5f}\n"
    )

    ax.legend()
    ax.set_title(text)
    plt.colorbar(s, ax=ax, label="threshold")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")

    ############################################################################
    # HISTOGRAM
    spacing = 51

    ax = axs[1]
    sub_df = df
    title = "Prediction histogram"

    preds = sub_df.preds
    targets = sub_df.targets
    targets = targets.astype(int)
    pos, bin = np.histogram(preds[targets == 1], np.linspace(0, 1, spacing))
    neg, bin = np.histogram(preds[targets == 0], np.linspace(0, 1, spacing))
    pos = pos / (targets == 1).sum()
    neg = neg / (targets == 0).sum()
    # plt.plot(bin[1:],neg, alpha=1)
    # plt.plot(bin[1:],pos, alpha=1)
    bin = (bin[1:] + bin[:-1]) / 2
    ax.bar(bin, neg, width=1 / spacing, label="neg", alpha=0.5)
    ax.bar(bin, pos, width=1 / spacing, label="pos", alpha=0.5)
    ax.legend()
    ax.set_title(title)

    # plt.show()
    plt.savefig(plot_save_path)


def print_metric(result, plot_save_path=None, df=None):

    print(f'{" ": <8}\t pfbeta\t auc\t |\t @th\t acc\t f1\t fbeta\t prec\t recall\t')

    reducers = ["single", "gbmean", "gbmax"]
    for name in reducers:
        text = f"{name: <8}"
        text += f'\t {result[f"{name}_pfbeta"]*100:0.2f}'
        text += f'\t {result[f"{name}_auc"]*100:0.2f}\t |'
        text += f'\t {result[f"{name}_best_thres"]:0.2f}'
        text += f'\t {result[f"{name}_best_acc"]*100:0.2f}'
        text += f'\t {result[f"{name}_best_f1score"]*100:0.2f}'
        text += f'\t {result[f"{name}_best_fbeta"]*100:0.2f}'
        text += f'\t {result[f"{name}_best_precision"]*100:0.2f}'
        text += f'\t {result[f"{name}_best_recall"]*100:0.2f}'
        # text += '\n'
        print(text)

    if plot_save_path is not None:
        print(f"Saving plot to {plot_save_path}")
        plot_pr_curve(result, plot_save_path, df)
