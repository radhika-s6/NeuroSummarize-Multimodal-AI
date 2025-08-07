import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from PIL import Image
import numpy as np
import os

# Hardcoded region-to-coordinate map (based on brain outline image)
BRAIN_REGION_COORDINATES = {
    "frontal lobe": (120, 80),
    "parietal lobe": (150, 50),
    "temporal lobe": (100, 150),
    "occipital lobe": (180, 130),
    "cerebellum": (140, 200),
    "hippocampus": (110, 170),
    "brainstem": (140, 230),
    "thalamus": (130, 120),
    "amygdala": (100, 180)
}

BRAIN_IMAGE_PATH = "assets/brain_diagram.png"  # Must exist relative to root

def show_affected_regions(region_list, image_path=BRAIN_IMAGE_PATH, save_path=None):
    """
    Overlay affected brain regions on a base anatomical image.

    Args:
        region_list (list of str): Brain regions to highlight.
        image_path (str): Path to the base brain image (PNG).
        save_path (str or None): If given, saves the output instead of displaying it.

    Returns:
        None
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at {image_path}")

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img)
    ax.axis("off")

    colors = plt.cm.get_cmap("Set1", len(region_list))

    for idx, region in enumerate(region_list):
        region_key = region.lower().strip()
        coord = BRAIN_REGION_COORDINATES.get(region_key)
        if coord:
            ax.plot(coord[0], coord[1], 'o', markersize=10, color=colors(idx))
            ax.text(coord[0] + 5, coord[1] - 5, region, fontsize=10, color=colors(idx), weight='bold')
        else:
            print(f"[WARN] Region not mapped: '{region}'")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_brain_heatmap(data):
    """
    Plot a heatmap showing frequency of affected regions.

    Args:
        data (list or dict): A list of regions (for count) or a dict of region -> score.

    Returns:
        fig: A Plotly bar chart figure.
    """
    if isinstance(data, list):
        region_counts = {}
        for region in data:
            region_key = region.lower().strip()
            region_counts[region_key] = region_counts.get(region_key, 0) + 1
    elif isinstance(data, dict):
        region_counts = {k.lower().strip(): v for k, v in data.items()}
    else:
        raise ValueError("[ERROR] Input must be list or dict.")

    regions = list(region_counts.keys())
    values = list(region_counts.values())

    fig = px.bar(
        x=regions,
        y=values,
        color=values,
        color_continuous_scale="Reds",
        labels={'x': 'Brain Region', 'y': 'Frequency'},
        title="Distribution of Affected Brain Regions"
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()
    return fig
