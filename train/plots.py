import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from base_plots import plot_loss

ALL_OUTPUT_NAMES = ["Expected", "Observed", "Expected Asimov", "Observed Asimov"]

def _output_names_for_cfg(cfg):
    target = list(cfg.data.target_indices) if cfg.data.get("target_indices") else list(range(4))
    return [ALL_OUTPUT_NAMES[i] for i in target]

def _subplot_grid(n):
    """Return (nrows, ncols) for n subplots, at most 2 columns."""
    if n == 1:
        return 1, 1
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    return nrows, ncols
# Replace or remove these lines:
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Charter"

# Instead, use sans-serif fonts (more likely available):
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

# Or if you want serif, try available ones:
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif"]

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = False
#plt.rcParams["text.latex.preamble"] = (
#    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"
#)

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, title, plot_dict):
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        if min(plot_dict["train_loss"]) < 0.0:
            logy = False
        else:
            logy = True
        plot_loss(
            file,
            [plot_dict["train_loss"], plot_dict["val_loss"]],
            plot_dict["train_lr"],
            labels=["train loss", "val loss"],
            logy=logy,
        )

    if cfg.plotting.histograms and cfg.evaluate:
        out = f"{plot_path}/histograms.pdf"
        out_names = _output_names_for_cfg(cfg)
        with PdfPages(out) as file:
            labels = ["Test", "Train", "Prediction"]

            for idataset, dataset in enumerate(cfg.data.dataset):
                data = [
                    plot_dict["results_test"][dataset]["raw"]["truth"].T,
                    plot_dict["results_train"][dataset]["raw"]["truth"].T,
                    plot_dict["results_test"][dataset]["raw"]["prediction"].T,
                ]
                print('range of data when plotting histograms:', [np.min(d) for d in data], [np.max(d) for d in data])
                plot_histograms_4outputs(
                    file, data, labels,
                    title=title[idataset], xlabel=r"$nLL$", logy=False,
                    output_names=out_names, plot_ratios=True,
                )
                if cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
                    labels = ["Test", "Train"]
                    sigmas_test = plot_dict["results_test"][dataset]["preprocessed"]["sigmas"].T
                    pull_test = plot_dict["results_test"][dataset]["preprocessed"]["pull"].T
                    sigmas_train = plot_dict["results_train"][dataset]["preprocessed"]["sigmas"].T
                    pull_train = plot_dict["results_train"][dataset]["preprocessed"]["pull"].T
                    sigmas = [sigmas_test, sigmas_train]
                    pulls = [pull_test, pull_train]
                    for sig_data, sig_labels, logy in [
                        (sigmas, labels, False),
                        ([sigmas_test], ["Test"], False),
                        ([sigmas_test], ["Test"], True),
                    ]:
                        plot_histograms_4outputs(
                            file, sig_data, sig_labels,
                            title=f'{title[idataset]} - $\sigma$', xlabel=r"$\sigma$",
                            logy=logy, output_names=out_names, plot_ratios=False,
                        )
                    for pull_data, pull_labels, logy in [
                        (pulls, labels, False),
                        ([pull_test], ["Test"], False),
                        ([pull_test], ["Test"], True),
                    ]:
                        plot_histograms_4outputs(
                            file, pull_data, pull_labels, xrange=(-5, 5),
                            title=f'{title[idataset]} - Pull',
                            xlabel=r"$\mathrm{pull} = \frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma}$",
                            logy=logy, output_names=out_names, plot_ratios=False, pull=True,
                        )

    if cfg.plotting.delta and cfg.evaluate:
        out_names = _output_names_for_cfg(cfg)
        # Scatter plots
        out = f"{plot_path}/scatter.pdf"
        with PdfPages(out) as file:
            for idataset, dataset in enumerate(cfg.data.dataset):
                truth_test = plot_dict["results_test"][dataset]["raw"]["truth"]
                pred_test = plot_dict["results_test"][dataset]["raw"]["prediction"]
                truth_train = plot_dict["results_train"][dataset]["raw"]["truth"]
                pred_train = plot_dict["results_train"][dataset]["raw"]["prediction"]

                n_out = truth_test.shape[1]
                nrows, ncols = _subplot_grid(n_out)
                fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows),
                                        gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
                axs_flat = np.array(axs).ravel()
                for i in range(n_out):
                    name = out_names[i] if i < len(out_names) else f"Output {i}"
                    plot_with_marginals(truth_test.T[i], pred_test.T[i],
                                        f'Δ Truth', f'Δ Pred', name, ax_main=axs_flat[i])
                for ax in axs_flat[n_out:]:
                    ax.set_visible(False)
                fig.suptitle(title[idataset], fontsize=16)
                fig.savefig(file, format="pdf", bbox_inches="tight")
                plt.close()

        # Standard delta plots
        out = f"{plot_path}/delta.pdf"
        with PdfPages(out) as file:
            for idataset, dataset in enumerate(cfg.data.dataset):
                truth_test = plot_dict["results_test"][dataset]["raw"]["truth"]
                pred_test = plot_dict["results_test"][dataset]["raw"]["prediction"]
                truth_train = plot_dict["results_train"][dataset]["raw"]["truth"]
                pred_train = plot_dict["results_train"][dataset]["raw"]["prediction"]

                delta_test = (pred_test - truth_test) / truth_test
                delta_train = (pred_train - truth_train) / truth_train

                xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]
                binss = [100, 50, 50]
                for xrange, bins in zip(xranges, binss):
                    plot_delta_histogram_4outputs(
                        file, [delta_test * 100, delta_train * 100],
                        labels=["Test", "Train"], title=title[idataset],
                        xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                        xrange=xrange, bins=bins, logy=False, mse_scale=1e-4,
                        output_names=out_names,
                    )
                    for logy in (False, True):
                        plot_delta_histogram_4outputs(
                            file, [delta_test * 100, delta_test * 100],
                            labels=["Test", "Largest 1\%"], title=title[idataset],
                            xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                            xrange=xrange, bins=bins, logy=logy, mse_scale=1e-4,
                            reference_truth=truth_test, output_names=out_names,
                        )

        # Absolute delta plots
        out = f"{plot_path}/delta_abs.pdf"
        with PdfPages(out) as file:
            for idataset, dataset in enumerate(cfg.data.dataset):
                truth_test = plot_dict["results_test"][dataset]["raw"]["truth"]
                pred_test = plot_dict["results_test"][dataset]["raw"]["prediction"]
                truth_train = plot_dict["results_train"][dataset]["raw"]["truth"]
                pred_train = plot_dict["results_train"][dataset]["raw"]["prediction"]

                delta_test = np.abs((pred_test - truth_test) / truth_test)
                delta_train = np.abs((pred_train - truth_train) / truth_train)

                xrange = (1e-8, 1)
                bins = 60
                plot_delta_histogram_4outputs(
                    file, [delta_test, delta_train], labels=["Test", "Train"],
                    title=title[idataset],
                    xlabel=r"$|\Delta| = |\frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}|$",
                    xrange=xrange, bins=bins, logx=True, logy=False, output_names=out_names,
                )
                plot_delta_histogram_4outputs(
                    file, [delta_test, delta_test], labels=["Test", "Largest 1\%"],
                    title=title[idataset],
                    xlabel=r"$|\Delta| = |\frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}|$",
                    xrange=xrange, bins=bins, logx=True, logy=False,
                    reference_truth=truth_test, output_names=out_names,
                )

        # Preprocessed delta plots
        if cfg.plotting.delta_prepd and cfg.evaluate:
            out = f"{plot_path}/delta_prepd.pdf"
            with PdfPages(out) as file:
                for idataset, dataset in enumerate(cfg.data.dataset):
                    truth_test = plot_dict["results_test"][dataset]["preprocessed"]["truth"]
                    pred_test = plot_dict["results_test"][dataset]["preprocessed"]["prediction"]
                    truth_train = plot_dict["results_train"][dataset]["preprocessed"]["truth"]
                    pred_train = plot_dict["results_train"][dataset]["preprocessed"]["prediction"]

                    delta_test = (pred_test - truth_test) / truth_test
                    delta_train = (pred_train - truth_train) / truth_train

                    xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]
                    binss = [100, 50, 50]
                    xlabel_prepd = r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]"
                    for xrange, bins in zip(xranges, binss):
                        plot_delta_histogram_4outputs(
                            file, [delta_test * 100, delta_train * 100],
                            labels=["Test", "Train"], title=title[idataset],
                            xlabel=xlabel_prepd, xrange=xrange, bins=bins,
                            logy=False, mse_scale=1e-4, output_names=out_names,
                        )
                        for logy in (False, True):
                            plot_delta_histogram_4outputs(
                                file, [delta_test * 100, delta_test * 100],
                                labels=["Test", "Largest 1\%"], title=title[idataset],
                                xlabel=xlabel_prepd, xrange=xrange, bins=bins,
                                logy=logy, mse_scale=1e-4,
                                reference_truth=truth_test, output_names=out_names,
                            )


def plot_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
):
    hists = []
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        axs[0].step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            axs[0].fill_between(
                bins,
                dup_last(hist) * scale,
                0.0 * dup_last(hist),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = np.divide(
            hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
        )  # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(
        0.04,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_delta_histogram(
    file,
    datas,
    labels,
    title,
    xrange,
    bins=60,
    xlabel=None,
    logy=False,
    logx=False,
):
    assert len(datas) == 2
    dup_last = lambda a: np.append(a, a[-1])
    if logx:
        bins = np.logspace(np.log(xrange[0]), np.log(xrange[1]), bins)
    else:
        _, bins = np.histogram(datas[0], bins=bins - 1, range=xrange)
    hists, scales, mses = [], [], []
    for data in datas:
        mse = np.mean(data**2)

        data = np.clip(data, xrange[0], xrange[1])
        hist, _ = np.histogram(data, bins=bins, range=xrange)
        scale = 1 / np.sum((bins[1:] - bins[:-1]) * hist)
        mses.append(mse)
        hists.append(hist)
        scales.append(scale)

    fig, axs = plt.subplots(figsize=(6, 4))
    for hist, scale, mse, label, color in zip(
        hists, scales, mses, labels, colors[1:3][::-1]
    ):
        axs.step(
            bins,
            dup_last(hist) * scale,
            color,
            where="post",
            label=label + r" ($\overline{\Delta^2} = {%.2g})$" % (mse * 1e-4),
        )  # need 1e-4 to compensate for initial *100
        axs.fill_between(
            bins,
            dup_last(hist) * scale,
            0.0 * dup_last(hist) * scale,
            facecolor=color,
            alpha=0.1,
            step="post",
        )

    if logy:
        axs.set_yscale("log")
    if logx:
        axs.set_xscale("log")
    ymin, ymax = axs.get_ylim()
    if not logy:
        ymin = 0.0
    axs.vlines(0.0, ymin, ymax, color="k", linestyle="--", lw=0.5)
    axs.set_ylim(ymin, ymax)
    axs.set_xlim(xrange)

    axs.set_xlabel(xlabel, fontsize=FONTSIZE)
    axs.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs.legend(frameon=False, loc="upper left", fontsize=FONTSIZE * 0.7)
    axs.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_histograms_4outputs(
    file,
    data,
    labels,
    n_bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
    output_names=None,
    plot_ratios=True,
    pull=False,
    reference_truth=None,
):
    # data: list of (n_outputs, N) arrays, one per dataset (test/train/pred)
    n_outputs = len(data[0])
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_outputs)]

    # Transpose: output_data[i] = tuple of per-dataset arrays for output i
    output_data = list(zip(*data))
    flat_reference = reference_truth

    nrows, ncols = _subplot_grid(n_outputs)

    if plot_ratios:
        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
        outer_gs = fig.add_gridspec(nrows, 1, hspace=0.35)
        main_axes, ratio_axes = [], []
        for row in range(nrows):
            inner = outer_gs[row].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0)
            for col in range(ncols):
                m = fig.add_subplot(inner[0, col])
                r = fig.add_subplot(inner[1, col], sharex=m)
                m.tick_params(labelbottom=False)
                main_axes.append(m)
                ratio_axes.append(r)
    else:
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        main_axes = np.array(axes_grid).ravel().tolist()
        ratio_axes = None

    for output_idx in range(n_outputs):
        # Get all datasets for this output
        datasets_for_output = list(output_data[output_idx])  # Make a copy we can modify
        
        # If reference truth provided and we have exactly 2 datasets, 
        # replace the second dataset with only the top 1% values
        if flat_reference is not None and len(datasets_for_output) == 2:
            truth_values = flat_reference[output_idx]
            largest_idx = round(0.01 * len(truth_values))
            sort_idx = np.argsort(truth_values)
            largest_min = truth_values[sort_idx][-largest_idx - 1]
            largest_mask = truth_values > largest_min
            datasets_for_output[1] = datasets_for_output[1][largest_mask]
        
        # Calculate bins based on all data (including modified data if reference_truth was used)
        if xrange is None:
            min_val = min(np.min(dat) for dat in datasets_for_output)
            max_val = max(np.max(dat) for dat in datasets_for_output)
            bins = np.linspace(min_val, max_val, n_bins + 1)
        else:
            bins = np.linspace(xrange[0], xrange[1], n_bins + 1)
        
        hists = []
        for dat in datasets_for_output:
            hist, _ = np.histogram(dat, bins=bins, range=xrange)
            hists.append(hist)
        
        integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
        scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
        dup_last = lambda a: np.append(a, a[-1])
        
        # Plot main histogram
        for i, hist, scale, label, color in zip(
            range(len(hists)), hists, scales, labels, colors
        ):
            main_axes[output_idx].step(
                bins,
                dup_last(hist) * scale,
                label=label,
                color=color,
                linewidth=1.0,
                where="post",
            )
            
            if i == 0:
                main_axes[output_idx].fill_between(
                    bins,
                    dup_last(hist) * scale,
                    0.0 * dup_last(hist),
                    facecolor=color,
                    alpha=0.1,
                    step="post",
                )
                continue
            
            # Plot ratio if enabled
            if plot_ratios:
                ratio = np.divide(
                    hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
                )
                ratio_axes[output_idx].step(
                    bins, dup_last(ratio), linewidth=1.0, where="post", color=color
                )
        
        # Configure axes
        if logx:
            main_axes[output_idx].set_xscale("log")
        if logy:
            main_axes[output_idx].set_yscale("log")
        if pull:
            # Overlay a standard Gaussian (0, 1) distribution
            lo, hi = xrange if xrange is not None else (-5, 5)
            x = np.linspace(lo, hi, 1000)
            gaussian = norm.pdf(x, 0, 1)
            main_axes[output_idx].plot(x, gaussian, 'r--', linewidth=2, label='Gaussian (0, 1)')
        
        main_axes[output_idx].set_title(output_names[output_idx], fontsize=FONTSIZE)
        main_axes[output_idx].tick_params(axis="both", labelsize=FONTSIZE_TICK)
        main_axes[output_idx].relim()
        main_axes[output_idx].autoscale_view()
        main_axes[output_idx].legend(loc="best", frameon=False, fontsize=FONTSIZE_LEGEND)

        if plot_ratios:
            ratio_axes[output_idx].set_yticks(ratio_ticks)
            ratio_axes[output_idx].set_ylim(ratio_range)
            ratio_axes[output_idx].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
            ratio_axes[output_idx].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
            ratio_axes[output_idx].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)
            ratio_axes[output_idx].tick_params(axis="both", labelsize=FONTSIZE_TICK)
            ratio_axes[output_idx].relim()
            ratio_axes[output_idx].autoscale_view()
    

    for ax in main_axes:
        ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    
    if plot_ratios:
        for ax in ratio_axes:
            ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    else:
        for ax in main_axes[:n_outputs]:
            ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    
    fig.suptitle(title, fontsize=FONTSIZE+2, y=1.02)
    fig.subplots_adjust(top=0.92)
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close()

def plot_delta_histogram_4outputs(
    file,
    datas,  # List of datasets, each containing multiple batches of shape (M,4)
    labels,
    title,
    xrange=None,
    bins=60,
    xlabel=None,
    logy=False,
    logx=False,
    output_names=None,
    mse_scale=1.0,
    reference_truth=None,
    pull=False
):
    assert len(datas) == 2

    n_outputs = datas[0].shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_outputs)]

    flat_datas = datas
    flat_reference = reference_truth
    if xrange is None:
        min_val = min(np.min(d) for d in flat_datas)
        max_val = max(np.max(d) for d in flat_datas)
        xrange = (min_val, max_val)

    nrows, ncols = _subplot_grid(n_outputs)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = np.array(axes_grid).ravel()
    dup_last = lambda a: np.append(a, a[-1])

    for output_idx in range(n_outputs):
        ax = axs[output_idx]
        
        # Get flattened data for this output
        output_data1 = flat_datas[0][:, output_idx]
        output_data2 = flat_datas[1][:, output_idx]
        
        # If reference truth provided, calculate largest 1% for this specific output
        if flat_reference is not None:
            truth_values = flat_reference[:, output_idx]
            largest_idx = round(0.01 * len(truth_values))
            sort_idx = np.argsort(truth_values)
            largest_min = truth_values[sort_idx][-largest_idx - 1]
            largest_mask = truth_values > largest_min
            output_data2 = output_data2[largest_mask]
        
        if logx:
            bin_edges = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), bins)
        else:
            _, bin_edges = np.histogram(output_data1, bins=bins-1, range=xrange)
            
        # Calculate histograms
        hist1, _ = np.histogram(output_data1, bins=bin_edges, range=xrange)
        hist2, _ = np.histogram(output_data2, bins=bin_edges, range=xrange)
        
        # Normalization
        scale1 = 1 / np.sum((bin_edges[1:] - bin_edges[:-1]) * hist1) if np.sum(hist1) > 0 else 1.0
        scale2 = 1 / np.sum((bin_edges[1:] - bin_edges[:-1]) * hist2) if np.sum(hist2) > 0 else 1.0
        
        # MSE calculation
        mse1 = np.mean(output_data1**2) if len(output_data1) > 0 else 0.0
        mse2 = np.mean(output_data2**2) if len(output_data2) > 0 else 0.0
        
        # Plotting
        for hist, scale, mse, label, color in zip(
            [hist1, hist2], [scale1, scale2], [mse1, mse2], labels, colors[1:3][::-1]
        ):
            if len(hist) > 0:  # Only plot if we have data
                ax.step(
                    bin_edges,
                    dup_last(hist) * scale,
                    color,
                    where="post",
                    label=label + r" ($\overline{\Delta^2} = {%.2g})$" % (mse * mse_scale),
                )
                ax.fill_between(
                    bin_edges,
                    dup_last(hist) * scale,
                    0.0 * dup_last(hist) * scale,
                    facecolor=color,
                    alpha=0.1,
                    step="post",
                )
        if pull:
            # Overlay a standard Gaussian (0, 1) distribution
            x = np.linspace(xrange[0], xrange[1], 1000)
            gaussian = norm.pdf(x, 0, 1)
            ax.plot(x, gaussian, 'r--', linewidth=2, label='Gaussian (0, 1)')
            
        # Subplot configuration
        if logy:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
            
        ymin, ymax = ax.get_ylim()
        if not logy:
            ymin = 0.0
        if not logx and xrange[0] <= 0 <= xrange[1]:
            ax.vlines(0.0, ymin, ymax, color="k", linestyle="--", lw=0.5)
        if pull:
            ax.set_ylim(bottom=0.9)
        else:
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(xrange)
        
        ax.set_title(output_names[output_idx], fontsize=FONTSIZE)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
        ax.legend(frameon=False, loc="upper left", fontsize=FONTSIZE*0.7)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_ylabel("Normalized", fontsize=FONTSIZE)

    for ax in axs[n_outputs:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=FONTSIZE+2, y=1.02)
    plt.tight_layout()
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_hist_pulls(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None
    ):
    # model, data='test', yscale='linear', yrange=False, range_hist=False, n_bins=30, exclude=False):
    # model_name = model.metadata.get('name', 'Unknown')  
    # dataset = model.metadata.get('dataset', 'Unknown') 
    # labels = ['Expected', 'Observed', 'Expected Asimov', 'Observed Asimov']
    
    # Retrieve pull values
    if exclude:
        pulls, pulls_high, pulls_low = get_pull(model, data, exclude=exclude)
    else:
        pulls = get_pull(model, data)
    
    # If no pull values are found, exit the function
    if pulls.size == 0:
        print(f"No pull values found for dataset '{data}'. Skipping plotting.")
        return
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()  # Flatten the axes array for easier iteration
    
    # Plot histograms for each pull type in separate subplots
    for i, ax in enumerate(axes):
        if i >= len(pulls):  # In case we have fewer than 4 pull types
            ax.axis('off')
            continue

        if type(range_hist)==list:
            range_histt=range_hist[i]
        else:
            range_histt=range_hist
        # Plot the histogram
        if exclude:
            plot_histogram(ax,[pulls[i],pulls_high[0][i],pulls_low[0][i]],[f'Pull',f'Values > {exclude}',f'Values < {exclude}'],range_histt,xlabel='Pull values')
        else:
            plot_single_histogram(ax, [pulls[i]], [f'Pull'], range_histt, xlabel='Pull values')
        
        # Overlay a standard Gaussian (0, 1) distribution
        x = np.linspace(range_hist[0], range_histt[1], 1000)
        gaussian = norm.pdf(x, 0, 1)
        ax.plot(x, gaussian, 'r--', linewidth=2, label='Gaussian (0, 1)')
        
        # Customize the subplot
        ax.set_yscale(yscale)
        if type(yrange)==list:
            y_range=yrange[i]
        else:
            y_range=yrange
        ax.set_ylim(y_range)
        ax.legend(frameon=False, fontsize=FONTSIZE)
        ax.set_title(labels[i], fontsize=FONTSIZE)
    
    plt.tight_layout()
    
    # Save the plot
    if exclude:
        fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pdfs/{model_name}_{data}_all_{yscale}_hist_pulls_exclude{exclude}.pdf')
        fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pngs/{model_name}_{data}_all_{yscale}_hist_pulls_exclude{exclude}.png')
    else:
        fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pdfs/{model_name}_{data}_all_{yscale}_hist_pulls.pdf')
        fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pngs/{model_name}_{data}_all_{yscale}_hist_pulls.png')
    
    plt.show()

def plot_hist_sigmas(models, labels_hist, data='test', logy=False, logx=False, yrange=False, range_hist=False, n_bins=60, colors='black'):
    if type(models)!=list:
        models=[models]
    if type(labels_hist)!=list:
        labels_hist=[labels_hist]
       
    sigmas=[]
    for model in models:
        model_name = model.metadata.get('name', 'Unknown')  
        dataset = model.metadata.get('dataset', 'Unknown') 
        labels = ['Expected', 'Observed', 'Expected Asimov', 'Observed Asimov']

        # Retrieve sigmas
        sigmass = get_sigmas(model, data)

        # If no sigmas values are found, exit the function
        if sigmass.size == 0:
            print(f"No sigmas values found for dataset '{data}'. Skipping plotting.")
            return
        sigmas.append(sigmass)
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()  # Flatten the axes array for easier iteration
    
    # Plot histograms for each sigmas type in separate subplots
    for i, ax in enumerate(axes):
        hists=[]
        for s in sigmas:
            hists.append(s[i])
        if type(range_hist)==list:
            range_histt=range_hist[i]
        else:
            range_histt=range_hist
        labelss_hist=labels_hist.copy()
        #for j in range(len(labelss_hist)):
        #    labelss_hist[j] = labelss_hist[j] + fr' $\sigma$ {labels[i]}'
        # Plot the histogram
        plot_single_histogram(ax, hists, labelss_hist, range_histt, xlabel=r'$\sigma$', bins=n_bins, logy=logy, logx=logx, colors=colors )
        
        # Customize the subplot
        #ax.set_yscale(yscale)
        if type(yrange)==list:
            y_range=yrange[i]
        else:
            y_range=yrange
        ax.set_ylim(y_range)
        ax.legend(frameon=False, fontsize=FONTSIZE)
        ax.set_title(labels[i], fontsize=FONTSIZE)
    
    plt.tight_layout()
    
    name_of_plot=f'{model_name}_{data}_all_hist_sigmas'
    if logy:
        name_of_plot+='_logy'
    if logx:
        name_of_plot+='_logx'
    # Save the plot
    fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pdfs/{name_of_plot}.pdf')
    fig.savefig(f'{path_home}/outputs/{dataset}/{model_name}/plots/pngs/{name_of_plot}.png')
    
    plt.show()

    
def plot_single_histogram(ax, datas, labels, xrange, xlabel=None, bins=60, logy=False, logx=False, colors=False):
    """Modified version of plot_histogram that works with existing axes"""
    # Ensure datas is a list
    if not isinstance(datas, list):
        datas = [datas]
    
    # Ensure labels is a list and has the same length as datas
    if not isinstance(labels, list):
        labels = [labels]
    if len(labels) != len(datas):
        raise ValueError("Length of labels must match the number of datasets in datas")
        
    dup_last = lambda a: np.append(a, a[-1])
    if logx:
        bins = np.logspace(np.log(xrange[0]), np.log(xrange[1]), bins)
    else:
        _, bins = np.histogram(datas[0], bins=bins - 1, range=xrange)

    hists, scales = [], []
    for data in datas:
        data = np.clip(data, xrange[0], xrange[1])
        hist, _ = np.histogram(data, bins=bins, range=xrange)
        scale = 1 / np.sum((bins[1:] - bins[:-1]) * hist)
        hists.append(hist)
        scales.append(scale)
       
    if colors and type(colors)!=list:
        colors=[colors]
    elif colors:
        pass
    # else:
    #     colors = 'black'
    for hist, scale, label, color in zip(hists, scales, labels, colors):
        ax.step(
            bins,
            dup_last(hist) * scale,
            color,
            where="post",
            label=label,
        )  
        ax.fill_between(
            bins,
            dup_last(hist) * scale,
            0.0 * dup_last(hist) * scale,
            facecolor=color,
            alpha=0.1,
            step="post",
        )

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ymin, ymax = ax.get_ylim()
    if not logy:
        ymin = 0.0
    ax.vlines(0.0, ymin, ymax, color="k", linestyle="--", lw=0.5)
    ax.set_xlim(xrange)

    ax.set_xlabel(xlabel)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.legend(frameon=False, loc="upper left", fontsize=FONTSIZE)

def plot_gradients(file, model, iteration):
    layer_names = []
    grad_values = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_values.append(param.grad.detach().cpu().view(-1).numpy())

    # Plotting
    n_layers = len(layer_names)
    if n_layers == 0:
        print("No gradients to plot.")
        return
        
    fig, axs = plt.subplots(n_layers, 1, figsize=(6, 2 * n_layers))

    if n_layers == 1:
        axs = [axs]

    for ax, name, grads in zip(axs, layer_names, grad_values):
        ax.hist(grads, range=(grads.min(),grads.max()), bins=50, alpha=0.7)
        ax.set_title(f'Gradient Histogram: {name}')
        ax.set_xlabel("Gradient value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(grads.min(), grads.max())
        print(name, grads.min(), grads.max(), grads.mean(), grads.std())
    plt.suptitle(f"Gradients at iteration {iteration}", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f'{file}/gradients_{iteration}.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def plot_with_marginals(y_truth, y_pred, label_x, label_y, title, ax_main, bins=None):
    """
    Plot scatter with stacked marginal histograms around each main axis.
    Assumes ax_main is provided; places horizontal marginal below and vertical to the right.
    Bins match symlog scale if not provided.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import SymmetricalLogLocator

    # Define APE group masks
    conditions = [
        (np.abs((y_truth - y_pred) / y_pred) <= 0.01),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.10),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.20),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.50),
        (np.abs((y_truth - y_pred) / y_pred) > 0.50),
    ]
    colors = ['cyan', 'green', 'orange', 'red', 'darkred']
    labels = [
        f"APE<=1% ({conditions[0].sum()})",
        f"1–10% ({(conditions[1]&~conditions[0]).sum()})",
        f"10–20% ({(conditions[2]&~conditions[1]).sum()})",
        f"20–50% ({(conditions[3]&~conditions[2]).sum()})",
        f">50% ({conditions[4].sum()})"
    ]

    # Get figure and position of main axis
    fig = ax_main.figure
    pos = ax_main.get_position()
    pad = 0.005
    hist_h = 0.12 * pos.height
    hist_w = 0.12 * pos.width

    # horizontal marginal BELOW main
    ax_histx = fig.add_axes([
        pos.x0,
        pos.y0 - pad - hist_h,
        pos.width,
        hist_h
    ])
    # vertical marginal RIGHT of main
    ax_histy = fig.add_axes([
        pos.x1 + pad,
        pos.y0,
        hist_w,
        pos.height
    ])

    # Main scatter
    mn, mx = np.min(y_truth), np.max(y_truth)
    ax_main.plot([mn, mx], [mn, mx], c='black', zorder=0)
    for color, label, mask in zip(colors, labels, [
        conditions[0],
        conditions[1] & ~conditions[0],
        conditions[2] & ~conditions[1],
        conditions[3] & ~conditions[2],
        conditions[4]
    ]):
        ax_main.scatter(
            y_truth[mask], y_pred[mask],
            s=20, alpha=0.6, c=color,
            marker='.', edgecolor='none',
            label=label
        )
    ax_main.tick_params(axis='x', which='both', labelbottom=False)
    # Symlog styling on main
    thresh = 0.01
    ax_main.set_xscale('symlog', linthresh=thresh)
    ax_main.set_yscale('symlog', linthresh=thresh)
    ax_main.set_xlabel(label_x)
    ax_main.set_ylabel(label_y)
    ax_main.set_title(title)
    ax_main.minorticks_on()
    ax_main.tick_params(which='both', direction='in', length=5)
    locator = SymmetricalLogLocator(base=10, linthresh=thresh,
                                    subs=np.arange(1, 10) * 0.1)
    ax_main.xaxis.set_minor_locator(locator)
    ax_main.yaxis.set_minor_locator(locator)
    ax_main.grid(which='both', linestyle='--', linewidth=0.25)
    ax_main.legend(fontsize='small')

    # Compute bins if not provided
    ax_main.relim()
    ax_main.autoscale_view()
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    bins_x = np.unique(locator.tick_values(*xlim))
    bins_y = np.unique(locator.tick_values(*ylim))

    # Align histogram axes limits to main
    ax_histx.set_xscale('symlog', linthresh=thresh)
    ax_histx.set_yscale('log')
    # ax_histx.set_xlim(ax_main.get_xlim())
    ax_histx.minorticks_on()
    ax_histx.hist([
        y_truth[conditions[0]],
        y_truth[conditions[1] & ~conditions[0]],
        y_truth[conditions[2] & ~conditions[1]],
        y_truth[conditions[3] & ~conditions[2]],
        y_truth[conditions[4]]
    ], bins=bins_x, stacked=True, color=colors, alpha=0.5, range=xlim)
    # ax_histx.axis('off')
    ax_histx.set_xlim(xlim)


    ax_histy.set_yscale('symlog', linthresh=thresh)
    ax_histy.set_xscale('log')
    # ax_histy.set_ylim(ax_main.get_ylim())
    ax_histy.minorticks_on()
    ax_histy.hist([
        y_pred[conditions[0]],
        y_pred[conditions[1] & ~conditions[0]],
        y_pred[conditions[2] & ~conditions[1]],
        y_pred[conditions[3] & ~conditions[2]],
        y_pred[conditions[4]]
    ], bins=bins_y, orientation='horizontal', stacked=True, color=colors, alpha=0.5, range=ylim)
    ax_histy.tick_params(axis='y', which='both', labelleft=False)
    ax_histy.set_ylim(ylim)


    return ax_main, ax_histx, ax_histy