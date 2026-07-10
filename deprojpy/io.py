from pathlib import Path

import labelimage_tools as lit
import numpy as np
import tifffile


def load_tiff_pair(
    mask_path: str | Path, heightmap_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DeProj-compatible segmentation mask and height map.

    Parameters
    ----------
    mask_path, heightmap_path:
        Paths to two-dimensional TIFF images. Array axes follow image
        convention ``(row, column)``.

    Returns
    -------
    mask, heightmap:
        NumPy arrays with identical shapes. Geometry produced later by
        :func:`deprojpy.from_heightmap` uses explicit ``(x, y, z)`` order.
    """
    mask = np.asarray(tifffile.imread(mask_path))
    heightmap = np.asarray(tifffile.imread(heightmap_path))
    if mask.ndim != 2 or heightmap.ndim != 2:
        raise ValueError("mask and heightmap must both be 2-D images")
    if mask.shape != heightmap.shape:
        raise ValueError(f"shape mismatch: mask {mask.shape}, heightmap {heightmap.shape}")
    return mask, heightmap


def load_label_heightmap_pair(
    labels_path: str | Path,
    heightmap_path: str | Path,
    *,
    background: int = 0,
    preprocess_labels: bool = True,
    crop_to_foreground: bool = False,
    remove_small_bits: bool = True,
    fill_holes: bool = True,
    dilate_borders: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    connectivity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a labeled-cell image and matching height map."""
    heightmap = np.asarray(tifffile.imread(heightmap_path))
    if preprocess_labels:
        labels = lit.load_image_pipeline(
            labels_path,
            seed=seed,
            background=background,
            connectivity=connectivity,
            crop_to_foreground=False,
            remove_small_bits=remove_small_bits,
            fill_holes=fill_holes,
            dilate_borders=dilate_borders,
            shuffle=shuffle,
        )
    else:
        labels = lit.load_label_image(labels_path)
    labels = np.asarray(labels)
    if labels.ndim != 2 or heightmap.ndim != 2:
        raise ValueError("labels and heightmap must both be 2-D images")
    if crop_to_foreground:
        labels, slices = lit.crop_to_foreground_bbox(labels, background=background, padding=5)
        heightmap = heightmap[slices]
    if labels.shape != heightmap.shape:
        raise ValueError(f"shape mismatch: labels {labels.shape}, heightmap {heightmap.shape}")
    return labels, heightmap
