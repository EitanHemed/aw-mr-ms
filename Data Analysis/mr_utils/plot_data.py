import functools
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from mr_utils import constants as cn

COLOR_HEX_VALUES = ['#BF40BF', '#228B22']
PALETTE = sns.color_palette(COLOR_HEX_VALUES)

HIGH_INCONCLUSIVE_LIMIT = 3
LOW_INCONCLUSIVE_LIMIT = 1 / 3
INCONCLUSIVENESS_REGION_LINESTYLE = '--'
INCONCLUSIVENESS_REGION_LINECOLOR = 'black'
INCONCLUSIVENESS_REGION_FACECOLOR = 'silver'
SEQUENTIAL_BAYES_LINE_STYLE = '-'
SEQUENTIAL_BAYES_LINE_WIDTH = 2

FIGURES_EXPORT_FORMATS = ['pdf', 'png']
FIGURES_FACECOLOR = 'white'
FIGURES_RESOLUTION = 600

np.random.seed(42)


def main(figs_dir: typing.Union[str, pathlib.Path], aggregated_data: pd.DataFrame, deltas_data: pd.DataFrame,
         sequential_bf_results: dict, task: pd.DataFrame) -> None:
    """Draws and saves the main figure of the paper.

    Parameters
    ----------

    figs_dir: str or pathlib.Path object where the figure will be saved.
    aggregated_data: A pandas DataFrame containing the aggregated data.
    deltas_data: A pandas DataFrame containing the data from the deltas RT measure.
    sequential_bf_results: A dictionary containing the results of the sequential Bayes factor analysis.
    task: A pandas DataFrame containing the data from the task, following preprocessing.

    Returns
    -------
    None
    """
    plot_data(aggregated_data, deltas_data, figs_dir, sequential_bf_results)

    plot_supplemental(figs_dir, task)


def plot_data(aggregated_data, deltas_data, figs_dir, sequential_bf_results):
    fig = plt.figure(constrained_layout=True, figsize=[9, 8.5])
    sub_figs = fig.subfigures(3, 1, hspace=0.05, height_ratios=[1, 10, 5])
    legend_subaxs = sub_figs[0].subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 10, 1, 1]}, )
    legend_ax = legend_subaxs[2]
    legend_ax.set_axis_off()
    [ax.remove() for ax in legend_subaxs[[0, 1, 3, 4]]]
    rain_cloud_axs = sub_figs[1].subplots(1, 5, gridspec_kw={'width_ratios': [2, 1.5, 4, 1.5, 2],
                                                             'wspace': 0.1},
                                          sharey=True, )
    draw_raincloud(aggregated_data, rain_cloud_axs, legend_ax)
    deltas_axs = sub_figs[2].subplots(1, 2, gridspec_kw={'width_ratios': [2, 4]})
    draw_deltas(deltas_data, deltas_axs, sequential_bf_results)
    fig.tight_layout(
        pad=0.05
    )

    for i in FIGURES_EXPORT_FORMATS:
        fig.savefig(os.path.join(figs_dir, f'main_raincloud_plot.{i}'),
                    dpi=FIGURES_RESOLUTION, facecolor=FIGURES_FACECOLOR,
                    )


def draw_raincloud(data: pd.DataFrame, rain_cloud_axs: np.ndarray, legend_ax: plt.Axes) -> None:
    """Draws the raincloud plot - a scatter plot, boxplot and KDE plot of the data.

    Parameters
    ----------
    data: A pandas DataFrame containing the data to plot.
    rain_cloud_axs: A numpy array of matplotlib Axes objects to plot the data on.
    legend_ax: A matplotlib Axes object to plot the legend on.

    Returns
    -------
    None
    """
    timepoints_grouper = cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE
    hue_grouper = cn.COLUMN_NAME_EXP_GROUP
    y = cn.COLUMN_NAME_MEAN_ROTATION_RT
    within_levels = data[timepoints_grouper].unique()

    assert within_levels.size == 2

    time_1_indices = (
            data[timepoints_grouper] == within_levels[0]).values

    sizes = data.loc[time_1_indices].groupby(hue_grouper).size().values
    new_labels = [f'{lab} (N = {n:.0f})' for lab, n in zip(cn.REPRESENTATIVE_GROUP_LABELS, sizes)]

    # Curried functions for plotting
    kde_func = functools.partial(sns.kdeplot, **dict(legend=False, cut=0, y=y, hue=hue_grouper,
                                                     palette=PALETTE, fill=True, alpha=.5, linewidth=0, ))
    box_func = functools.partial(sns.boxplot, **dict(x=hue_grouper, y=y, hue=hue_grouper, palette=PALETTE, ))

    # Add KDEs
    kde_left = kde_func(data=data.loc[time_1_indices], ax=rain_cloud_axs[0], )
    box_left = box_func(data=data.loc[time_1_indices], ax=rain_cloud_axs[1], )

    # Add legend
    handles, _ = box_left.get_legend_handles_labels()
    box_left.get_legend().remove()
    legend_ax.legend(handles, new_labels, loc='center', frameon=False, fontsize=14, ncols=2)

    box_right = box_func(data=data.loc[~time_1_indices], ax=rain_cloud_axs[3], )
    box_right.get_legend().remove()
    kde_right = kde_func(data=data.loc[~time_1_indices], ax=rain_cloud_axs[4], )

    # Add scatterplot of individual data points
    for (name, group), offset_sign, color in zip(data.groupby(hue_grouper), [-1, 1], PALETTE):
        for subject, subject_data in group.groupby(cn.COLUMN_NAME_UID):
            # Offset x values by a random amount, to avoid overlap
            x_vals = np.array([0, 1]) + 0.075 * offset_sign + np.random.uniform(-0.1, 0.1, size=1)
            y_vals = subject_data.groupby(timepoints_grouper)[y].mean().values.flatten()

            rain_cloud_axs[2].plot(x_vals, y_vals, color=color, alpha=0.25, linewidth=3, label=None, marker='o',
                                   markersize=1, )

    sns.pointplot(data=data, ax=rain_cloud_axs[2], y=y, hue=hue_grouper, palette=PALETTE, x=timepoints_grouper,
                  capsize=.075, errwidth=1.5, dodge=.15, join=False, markers='_')

    for (rot_name, rot_group), ax in zip(data.groupby(timepoints_grouper), rain_cloud_axs[[1, 3]]):
        # Annotate each plot using descriptive statistics
        for (chroma_name, chroma_group), _color, loc in zip(rot_group.groupby(
                cn.COLUMN_NAME_EXP_GROUP), PALETTE, (0.1, 0.9)):
            stats = chroma_group[y].agg(['mean', 'median', np.std]).values.round(2)
            xtick_label = '{}\n{}\n{}'.format(*stats)
            ax.annotate(xy=[loc, -0.075], ha='center', va='center', color=_color,
                        text=xtick_label, xycoords='axes fraction')

        # Set titles
        ax.set_title(f'|{rot_name}|°', loc='center', fontweight='bold', fontsize=12)

    # Aesthetics and customizations
    [rain_cloud_axs[2].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
    rain_cloud_axs[2].legend().remove()
    rain_cloud_axs[2].set(
        xticklabels=[f'|{float(i.get_text()):.0f}|°' for i in rain_cloud_axs[2].get_xticklabels()],
        xlabel=cn.VARIABLES_REPRESENTATIVE_NAMES[timepoints_grouper], ylabel='',
        yticklabels=[])

    rain_cloud_axs[4].invert_xaxis()

    [ax.set_axis_off() for ax in rain_cloud_axs[[1, 3, 4]]]

    [rain_cloud_axs[0].spines[spine].set_visible(False) for spine in ['top', 'right']]
    rain_cloud_axs[0].set(
        yticks=range(0, int(round(data[y].max(), -3) + 1000), 1000),
        yticklabels=range(0, int(round(data[y].max(), -3) + 1000), 1000),
        xlabel='Density', ylabel=cn.VARIABLES_REPRESENTATIVE_NAMES[y], xticks=[])


def draw_deltas(deltas_data: pd.DataFrame, deltas_axs: np.ndarray, sequential_bf_results: dict) -> None:
    """Draws the wingplot and sequential plot of the RT deltas data.

    Parameters
    ----------
    deltas_data : pd.DataFrame
        The data to plot.
    deltas_axs : np.ndarray
        The axes to plot on.
    sequential_bf_results : dict
        The results of the sequential bayesian analysis.

    Returns
    -------
    None
    """
    wingplot_ax, sequential_ax = deltas_axs

    draw_wingplot(wingplot_ax, deltas_data)
    draw_sequential_plot(sequential_ax, sequential_bf_results)


def draw_wingplot(wingplot_ax: plt.Axes, deltas_data: pd.DataFrame) -> None:
    """Draws the wingplot of the RT deltas data.

    A Wingplot is a scatterplot of individual data points, with CI depicting the mean and 95% CI of the data.
    The scatter is jittered to avoid overlap, by sorting the data and plotting it in ascending order, creating
    two 'wings' flanking the CI.
    """
    for (name, group), pos, _color in zip(deltas_data.groupby(cn.COLUMN_NAME_EXP_GROUP), [0, 1], PALETTE):
        wingplot_ax.scatter(np.linspace(pos - 0.4, pos + 0.4, group.shape[0]),
                            np.sort(group[cn.COLUMN_NAME_ROTATION_DELTA].values),
                            alpha=0.4, color=_color, label=None, s=10, marker='_', )

    deltas_data = deltas_data.copy().sort_values(cn.COLUMN_NAME_EXP_GROUP)

    sns.pointplot(data=deltas_data, x=cn.COLUMN_NAME_EXP_GROUP, y=cn.COLUMN_NAME_ROTATION_DELTA,
                  # hue=cn.COLUMN_NAME_EXP_GROUP,
                  # palette=PALETTE,
                  color='k', capsize=.05, errwidth=1, join=False, ax=wingplot_ax, markers='_')

    wingplot_ax.axhline(0, color='k', lw=2, ls='dotted')
    wingplot_ax.set(ylabel='Δ Rotation Time [$\mathbf{_{|180|°}}$ - $\mathbf{_{|90|°}}]$\n(MS, 95% CI)', xlabel=None)
    wingplot_ax.legend().remove()


def draw_sequential_plot(sequential_ax: plt.Axes, sequential_bf_results: dict) -> None:
    max_group_size = sequential_bf_results['between_groups']['values'].size

    rect = Rectangle(
        (0, LOW_INCONCLUSIVE_LIMIT), max_group_size + 4,
                                     HIGH_INCONCLUSIVE_LIMIT - LOW_INCONCLUSIVE_LIMIT,
        linewidth=1, edgecolor=None,
        facecolor=INCONCLUSIVENESS_REGION_FACECOLOR)
    sequential_ax.add_patch(rect)

    sequential_ax.axhline(LOW_INCONCLUSIVE_LIMIT,
                          ls=INCONCLUSIVENESS_REGION_LINESTYLE,
                          color=INCONCLUSIVENESS_REGION_LINECOLOR)
    sequential_ax.axhline(HIGH_INCONCLUSIVE_LIMIT,
                          ls=INCONCLUSIVENESS_REGION_LINESTYLE,
                          color=INCONCLUSIVENESS_REGION_LINECOLOR)

    for (name, bfs), _color in zip(sequential_bf_results['within_groups'].items(), PALETTE):

        sequential_ax.plot(np.arange(len(bfs)), bfs, color=_color, lw=SEQUENTIAL_BAYES_LINE_WIDTH,
                           ls=SEQUENTIAL_BAYES_LINE_STYLE, label=None)

        sequential_ax.scatter(len(bfs), bfs[-1], facecolors='none', edgecolors=_color, s=50,
                              label=f'{name}\n{bfs[-1]:.2E}')

        leg = sequential_ax.legend(title='$\mathbf{BF_{1:0}}$',
                                   # fontsize=LEGEND_ENTRY_FONTSIZE,
                                   columnspacing=0.2, labelspacing=0.1, loc='upper right', handletextpad=-0.2,
                                   handlelength=0, markerscale=0,  # 1.5,
                                   borderpad=0.05, borderaxespad=0.05, frameon=False,
                                   )

        for _color, text in zip(PALETTE, leg.get_texts()):
            text.set_color(_color)

        sequential_ax.set(xlim=[0, max_group_size + 4], yscale='log')

    for group, _color in zip(['x', 'y'], PALETTE):
        x_locs = np.where(sequential_bf_results['between_groups']['group_labels'] == group)
        sequential_ax.scatter(x_locs, sequential_bf_results['between_groups']['values'][x_locs],
                              color=_color, label=None, s=10, marker="_")

    sequential_ax.plot(
        sequential_bf_results['between_groups']['values'].size,
        sequential_bf_results['between_groups']['values'][-1],
        fillstyle='top', marker='o', markersize=10,
        markerfacecolor=PALETTE[0], markerfacecoloralt=PALETTE[1], markeredgecolor='k', linestyle='None',
        label='{} vs. \n{}\n{:.2E}'.format(
            *cn.REPRESENTATIVE_GROUP_LABELS, sequential_bf_results['between_groups']['values'][-1],
        )
    )

    sequential_ax.set(ylabel='Bayes Factor 1:0 (log-scale)', xlabel='Participants (N)')

    sequential_ax.legend()


def plot_supplemental(figures_dir: typing.Union[str, os.PathLike], task_data: pd.DataFrame) -> None:
    """Plot group average RTs for each rotation size and group, across the experiment"""

    g = sns.FacetGrid(task_data, col=cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,
                      # row=cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,
                      margin_titles=True, hue=cn.COLUMN_NAME_EXP_GROUP,
                      hue_kws={'color': sns.color_palette(['#BF40BF', '#228B22'])}, )

    g.figure.subplots_adjust(wspace=0, hspace=0)

    g.set_titles(col_template='|{col_name}|°')

    # Plot the mean smoothed line
    g.map(sns.lineplot, cn.COLUMN_NAME_TRIAL_NUM, cn.COLUMN_NAME_RAW_ROTATION_RT,
          errorbar=('ci', 95), err_style="band", linewidth=2,
          )

    [ax.set(xlabel='') for ax in g.axes[:, 1]]
    [ax.set(ylabel='') for ax in g.axes[0, :]]

    g.axes[0, 0].set_ylabel(cn.VARIABLES_REPRESENTATIVE_NAMES[cn.COLUMN_NAME_RAW_ROTATION_RT])
    g.axes[0, 0].set_xlabel(cn.VARIABLES_REPRESENTATIVE_NAMES[cn.COLUMN_NAME_TRIAL_NUM])

    g.axes[0, 0].legend(loc='upper right')

    for i in FIGURES_EXPORT_FORMATS:
        g.fig.savefig(os.path.join(figures_dir, f'supplemental.{i}'), dpi=FIGURES_RESOLUTION, bbox_inches='tight',
                      facecolor=FIGURES_FACECOLOR)
