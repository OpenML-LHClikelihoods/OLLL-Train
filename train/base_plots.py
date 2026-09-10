import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12


def plot_loss(file, losses, lr=None, labels=None, logy=True):
    if len(losses[1]) == 0:  # catch no-validations case
        losses = [losses[0]]
        labels = [labels[0]]
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0]) + 1)
    fig, ax = plt.subplots()
    min_val = min(min(sublist) for sublist in losses)
    if min_val<0:
        for i in range(len(losses)):
            losses[i] = [x - min_val - 1/min_val for x in losses[i]]
    for i, loss, label in zip(range(len(losses)), losses, labels):
        if len(loss) == len(iterations):
            its = iterations
        else:
            frac = len(losses[0]) / len(loss)
            its = np.arange(1, len(loss) + 1) * frac
        ax.plot(its, loss, label=label)
    #    logy=False
    if logy:
        ax.set_yscale("log")
    #ax.set_xscale("log")
    if lr is not None:
        axright = ax.twinx()
        axright.plot(iterations, lr, label="learning rate", color="crimson")
        axright.set_ylabel("Learning rate", fontsize=FONTSIZE)

    #ax.set_ylim(top=min(10,max(losses[0])),bottom=min(min(losses)))
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metric(file, metrics, metric_label, labels=None, logy=False):
    labels = [None for _ in range(len(metrics))] if labels is None else labels
    iterations = range(1, len(metrics[0]) + 1)
    fig, ax = plt.subplots()
    for i, metric, label in zip(range(len(metrics)), metrics, labels):
        if len(metric) == len(iterations):
            its = iterations
        else:
            frac = len(metrics[0]) / len(metric)
            its = np.arange(1, len(metric) + 1) * frac
        ax.plot(its, metric, label=label)

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric_label, fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper left")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()