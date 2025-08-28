import os
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting, image

# Download atlas once; cached afterwards
_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = image.load_img(_atlas.maps)
labels = [lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl for lbl in _atlas.labels]

def _match_labels(requested: List[str]) -> List[int]:
    req_lower = [r.lower() for r in requested]
    idxs = []
    for i, lbl in enumerate(labels):
        if i == 0:  # background
            continue
        l = lbl.lower()
        if any(r in l for r in req_lower):
            idxs.append(i)
    return idxs

def make_region_overlay(regions: List[str], out_png: str):
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    idxs = _match_labels(regions)
    if not idxs:
        # still render the atlas outline
        display = plotting.plot_stat_map(atlas_img, display_mode='ortho', draw_cross=False, annotate=False)
        display.savefig(out_png, dpi=150)
        display.close()
        return

    mask_data = np.isin(image.get_data(atlas_img), idxs).astype(int)
    mask_img = image.new_img_like(atlas_img, mask_data)

    display = plotting.plot_roi(mask_img, bg_img=None, display_mode='ortho', draw_cross=False, annotate=False)
    display.savefig(out_png, dpi=150)
    display.close()

# Load Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = nib.load(atlas.filename)
atlas_labels = atlas.labels

def map_terms_to_mask(terms: list):
    atlas_data = atlas_img.get_fdata()
    mask_data = np.zeros_like(atlas_data)
    for term in terms:
        if term in atlas_labels:
            idx = atlas_labels.index(term)
            mask_data[atlas_data == idx] = 1
    mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine)
    return mask_img

def show_affected_regions(terms: list, template_img=None):
    mask_img = map_terms_to_mask(terms)
    if template_img:
        mask_img = image.resample_to_img(mask_img, template_img)
    plotting.plot_stat_map(
        mask_img,
        title="Brain Region Map (best-effort)",
        display_mode='ortho',
        cut_coords=(0,0,0),
        colorbar=True
    )
    plotting.show()