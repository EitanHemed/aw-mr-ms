import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from mr_utils.plot_data import FIGURES_FACECOLOR, FIGURES_RESOLUTION, FIGURES_EXPORT_FORMATS, PALETTE

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16


# BAR_FILL_COLORS = sns.color_palette(np.repeat(
#     ['#BF40BF', '#228B22'], 2))


class StackedBarPlot:
    width = 0.35

    def __init__(self, data1, data2, labels1, labels2):
        self.data1 = data1
        self.data2 = data2
        self.labels1 = labels1
        self.labels2 = labels2
        self.num_groups = len(labels1)
        self.num_bars = len(data1)

    def create_plot(self) -> plt.Figure:

        fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

        self._add_stacked_bars(axes[0], self.data1, self.labels1, 'Group 1', xnnotations=['Monochrome', 'Multichrome'])
        self._add_stacked_bars(axes[1], self.data2, self.labels2, 'Group 2', xnnotations=['Monochrome', 'Multichrome'])

        axes[0].set_title('Possibility A', fontsize=BIGGER_SIZE, weight='bold')
        axes[1].set_title('Possibility B', fontsize=BIGGER_SIZE, weight='bold')

        axes[0].set_ylabel('Rotation Stage\n Response Time (ms)', fontsize=MEDIUM_SIZE, weight='bold')

        legend_handles, legend_labels = axes[1].get_legend_handles_labels()

        # Turn the first legend handle, into a hatch pattern, composed of two colors
        pa1 = Patch(facecolor=PALETTE[0], edgecolor='black', hatch='//')
        pa2 = Patch(facecolor=PALETTE[1], edgecolor='black', hatch='//')
        pb1 = Patch(facecolor='white', edgecolor='black', hatch='//')
        pb2 = Patch(facecolor='white', edgecolor='black', hatch='//')

        axes[0].legend(handles=[pa1, pb1, pa2, pb2, ],
                       labels=['', '', 'Rotation', 'Additional Process'],
                       ncol=2, handletextpad=0.5, handlelength=1, columnspacing=-0.5,
                       fontsize=MEDIUM_SIZE, loc='upper left'),

        axes[0].set_ylim(600, 2000)

        plt.tight_layout(h_pad=2, w_pad=2)

        return fig


    def _add_stacked_bars(self, ax, data, labels, group_name, xnnotations=None):
        num_bars = len(data)

        x = np.arange(self.num_groups)

        x = x.reshape(-1, 2)
        x += np.arange(x.shape[0]).reshape(-1, 1)
        x = x.flatten()

        bottom = np.zeros(self.num_groups)

        ax.bar(x, data[0], self.width, color=PALETTE,
               bottom=bottom, label='Rotation task', linewidth=2,
               hatch='//', edgecolor='black')
        ax.bar(x, data[1], self.width, color='white', bottom=data[0],
               label='Additional task', linewidth=2, hatch='//',
               edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=MEDIUM_SIZE)

        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Add annotation below the x axis, if provided - between every two bars
        if xnnotations is not None:

            bar_centers = x.reshape(-1, 2).mean(axis=1)  # x + self.width / 2
            for i in range(len(bar_centers)):
                ax.text(bar_centers[i], 400,
                        xnnotations[i], ha='center', va='bottom',
                        fontsize=BIGGER_SIZE, color='black', weight='bold')


def main(save_img_path: typing.Union[str, os.PathLike]) -> None:
    # Sample data and labels
    data1 = [[1000, 1300, 1000, 1300], [100, 100, 200, 200]]
    data2 = [[1000, 1300, 1100, 1500], [100, 100, 200, 200]]

    labels1 = ['90', '180', '90', '180']
    labels1 = [i + '°' for i in labels1]
    labels2 = [i for i in labels1]

    plotter = StackedBarPlot(data1, data2, labels1, labels2)
    fig = plotter.create_plot()

    for i in FIGURES_EXPORT_FORMATS:
        fig.savefig(os.path.join(save_img_path, f'illustration.{i}'),
                dpi=FIGURES_RESOLUTION, bbox_inches='tight', facecolor=FIGURES_FACECOLOR)


