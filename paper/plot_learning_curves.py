"""Generate publication-quality learning curves figure.

Produces a PDF and PNG of Dice coefficient by training set size,
with separate lines for liver, malignant mass, and benign mass.
Uses the unified color palette matching prediction overlay figures.

Usage:
    pip install matplotlib
    python paper/plot_learning_curves.py

Output:
    paper/learning_curves.pdf
    paper/learning_curves.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 10,
    'font.weight': '400',
    'axes.labelsize': 10,       # match tick/legend size
    'axes.labelcolor': '#999999',  # lighter gray for axis labels
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#999999',      # lighter gray for tick labels
    'ytick.color': '#999999',
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#e0e0e0',
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': True,
    'axes.spines.right': True,
})

BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#217a63'
# GRAY = '#898781'

sizes = [25, 50, 100, 200, 300, 400, 500, 625]
size_labels = ['25', '50', '100', '200', '300', '400', '500', '625']
liver = [0.820, 0.846, 0.878, 0.893, 0.898, 0.889, 0.899, 0.899]
malignant = [0.349, 0.455, 0.618, 0.654, 0.671, 0.701, 0.787, 0.767]
benign = [0.169, 0.147, 0.264, 0.311, 0.317, 0.341, 0.319, 0.403]

x = np.arange(len(sizes))

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, liver, color=BLUE, marker='o', markersize=7,
        linewidth=1.5, label='Liver', zorder=3,
        markeredgecolor='white', markeredgewidth=1.5)
ax.plot(x, malignant, color=ORANGE, marker='o', markersize=7,
        linewidth=1.5, label='Malignant mass', zorder=3,
        markeredgecolor='white', markeredgewidth=1.5)
ax.plot(x, benign, color=GREEN, marker='o', markersize=7,
        linewidth=1.5, linestyle='--', label='Benign mass', zorder=3,
        markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlabel('Training images')
ax.set_ylabel('Dice')
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(size_labels)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
ax.grid(False, axis='x')



ax.spines['bottom'].set_color('#c0c0c0')
ax.spines['left'].set_color('#c0c0c0')
ax.spines['top'].set_color('#c0c0c0')
ax.spines['right'].set_color('#c0c0c0')
ax.tick_params(axis='both', which='both',
               color='#c0c0c0',        # tick mark color (match grid/spine)
               labelcolor='#999999')   # tick label text color

ax.legend(loc='lower right', frameon=True, fancybox=False,
          borderpad=0.8, handlelength=2.5)

plt.tight_layout(pad=1.2)

out_dir = os.path.dirname(os.path.abspath(__file__))

fig.savefig(os.path.join(out_dir, 'learning_curves.pdf'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'learning_curves.png'),
            dpi=300, bbox_inches='tight')

print(f"Saved to {out_dir}/learning_curves.pdf and .png")
