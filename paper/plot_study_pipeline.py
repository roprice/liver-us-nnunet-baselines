"""Generate study pipeline schematic as PDF and PNG.

Edit the BOXES and EDGES lists below to adjust layout, colors, and text.
Coordinates are in figure units (0-14 x, 0-7 y). Origin is bottom-left.

Usage:
    python paper/plot_study_pipeline.py

Output:
    paper/study_pipeline_schematic.pdf
    paper/study_pipeline_schematic.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

FONT_FAMILY = ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
ARROW_COLOR = '#898781'
ARROW_LINEWIDTH = 1.2
LABEL_COLOR = '#52514e'   # color for edge labels like "110 test images"
CORNER_RADIUS = 0.15

# ---------------------------------------------------------------------------
# BOXES — each box is a dict with:
#   id:        unique name (used by EDGES)
#   center:    (x, y) position in figure coordinates
#   size:      (width, height)
#   fill:      background color
#   border:    border color
#   title:     bold top line
#   subtitle:  lighter lines below title (list of strings)
#   title_color:    color for the title text
#   subtitle_color: color for the subtitle text
# ---------------------------------------------------------------------------

BOXES = [
    {
        'id': 'dataset',
        'center': (1.8, 5.0),
        'size': (2.2, 1.2),
        'fill': '#f1efe8',
        'border': '#898781',
        'title': 'AUL dataset',
        'subtitle': ['735 images'],
        'title_color': '#52514e',
        'subtitle_color': '#898781',
    },
    {
        'id': 'split',
        'center': (5.2, 5.0),
        'size': (2.4, 1.8),
        'fill': '#e1f5ee',
        'border': '#0f6e56',
        'title': 'Stratified split',
        'subtitle': ['625 train', '110 test (fixed)', 'seed 42'],
        'title_color': '#085041',
        'subtitle_color': '#0f6e56',
    },
    {
        'id': 'subsets',
        'center': (5.2, 2.4),
        'size': (2.4, 1.6),
        'fill': '#eeedfe',
        'border': '#534ab7',
        'title': 'Nested subsets',
        'subtitle': ['25, 50, 100, 200', '300, 400, 500, 625'],
        'title_color': '#3c3489',
        'subtitle_color': '#534ab7',
    },
    {
        'id': 'nnunet',
        'center': (9.0, 3.5),
        'size': (2.2, 1.2),
        'fill': '#faece7',
        'border': '#eb6834',
        'title': 'nnU-Net',
        'subtitle': ['2D, 150 epochs'],
        'title_color': '#712b13',
        'subtitle_color': '#eb6834',
    },
    {
        'id': 'predictions',
        'center': (12.0, 3.5),
        'size': (1.8, 1.2),
        'fill': '#e6f1fb',
        'border': '#185fa5',
        'title': 'Predictions',
        'subtitle': ['8 models'],
        'title_color': '#0c447c',
        'subtitle_color': '#185fa5',
    },
    {
        'id': 'evaluation',
        'center': (11.2, 1.2),
        'size': (2.6, 1.2),
        'fill': '#eaf3de',
        'border': '#3b6d11',
        'title': 'Evaluation',
        'subtitle': ['Dice, detection rate'],
        'title_color': '#27500a',
        'subtitle_color': '#3b6d11',
    },
]

# ---------------------------------------------------------------------------
# EDGES — each edge is a dict with:
#   from:      id of source box
#   to:        id of target box
#   from_side: 'right', 'left', 'top', 'bottom' — which side the arrow leaves
#   to_side:   'right', 'left', 'top', 'bottom' — which side the arrow enters
#   label:     optional text label on the arrow
#   label_side: 'above' or 'below' the midpoint (default: 'above')
#   waypoints: optional list of (x, y) to route the arrow through
# ---------------------------------------------------------------------------

EDGES = [
    {
        'from': 'dataset',
        'to': 'split',
        'from_side': 'right',
        'to_side': 'left',
    },
    {
        'from': 'split',
        'to': 'subsets',
        'from_side': 'bottom',
        'to_side': 'top',
    },
    {
        'from': 'split',
        'to': 'nnunet',
        'from_side': 'right',
        'to_side': 'top',
        'label': '110 test images',
        'label_side': 'above',
        'waypoints': [(9.0, 5.0)],
    },
    {
        'from': 'subsets',
        'to': 'nnunet',
        'from_side': 'right',
        'to_side': 'left',
    },
    {
        'from': 'nnunet',
        'to': 'predictions',
        'from_side': 'right',
        'to_side': 'left',
    },
    {
        'from': 'nnunet',
        'to': 'evaluation',
        'from_side': 'bottom',
        'to_side': 'left',
        'label': 'vs ground truth',
        'label_side': 'left',
        'waypoints': [(9.0, 1.2)],
    },
    {
        'from': 'predictions',
        'to': 'evaluation',
        'from_side': 'bottom',
        'to_side': 'right',
        'waypoints': [(12.0, 1.2)],
    },
]


# ---------------------------------------------------------------------------
# RENDERING — you probably don't need to edit below here
# ---------------------------------------------------------------------------

def get_anchor(box, side):
    """Return (x, y) for the anchor point on a given side of a box."""
    cx, cy = box['center']
    w, h = box['size']
    if side == 'right':
        return (cx + w / 2, cy)
    elif side == 'left':
        return (cx - w / 2, cy)
    elif side == 'top':
        return (cx, cy + h / 2)
    elif side == 'bottom':
        return (cx, cy - h / 2)


def draw_box(ax, box):
    cx, cy = box['center']
    w, h = box['size']
    x0 = cx - w / 2
    y0 = cy - h / 2

    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
        facecolor=box['fill'],
        edgecolor=box['border'],
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(rect)

    # Title
    ax.text(cx, cy + 0.15 * (len(box['subtitle'])),
            box['title'],
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=box['title_color'],
            family=FONT_FAMILY[0],
            zorder=3)

    # Subtitle lines
    for i, line in enumerate(box['subtitle']):
        ax.text(cx, cy + 0.15 * (len(box['subtitle'])) - 0.32 * (i + 1),
                line,
                ha='center', va='center',
                fontsize=10, fontweight='normal',
                color=box['subtitle_color'],
                family=FONT_FAMILY[0],
                zorder=3)


def draw_edge(ax, edge, box_lookup):
    src = box_lookup[edge['from']]
    dst = box_lookup[edge['to']]

    start = get_anchor(src, edge['from_side'])
    end = get_anchor(dst, edge['to_side'])
    waypoints = edge.get('waypoints', [])

    # Build list of points
    points = [start] + waypoints + [end]

    # Draw line segments (all but last without arrow)
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        is_last = (i == len(points) - 2)

        ax.annotate(
            '',
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->' if is_last else '-',
                color=ARROW_COLOR,
                lw=ARROW_LINEWIDTH,
                shrinkA=0,
                shrinkB=3 if is_last else 0,
            ),
            zorder=1,
        )

    # Label
    label = edge.get('label')
    if label:
        label_side = edge.get('label_side', 'above')
        # Place label near the midpoint of the first segment
        if waypoints:
            mx = (start[0] + waypoints[0][0]) / 2
            my = (start[1] + waypoints[0][1]) / 2
        else:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2

        if label_side == 'above':
            my += 0.2
        elif label_side == 'below':
            my -= 0.25
        elif label_side == 'left':
            mx -= 0.15
            ha = 'right'
        elif label_side == 'right':
            mx += 0.15
            ha = 'left'

        ha = 'center'
        if label_side in ('left',):
            ha = 'right'
        elif label_side in ('right',):
            ha = 'left'

        ax.text(mx, my, label,
                ha=ha, va='center',
                fontsize=9, color=LABEL_COLOR,
                family=FONT_FAMILY[0],
                zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    box_lookup = {b['id']: b for b in BOXES}

    for box in BOXES:
        draw_box(ax, box)

    for edge in EDGES:
        draw_edge(ax, edge, box_lookup)

    plt.tight_layout(pad=0.5)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, 'study_pipeline_schematic.pdf'),
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out_dir, 'study_pipeline_schematic.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to {out_dir}/study_pipeline_schematic.{{pdf,png}}")


if __name__ == '__main__':
    main()
