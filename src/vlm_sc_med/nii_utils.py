from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _to_uint8(img: np.ndarray, clip_percentiles: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    lo, hi = np.percentile(img, clip_percentiles)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return (img * 255.0).astype(np.uint8)


def nifti_center_slices_to_mosaic_png(
    nii_path: str,
    out_png: str,
    *,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    max_side: int = 512,
) -> str:
    """
    Create a 2D mosaic PNG from a 3D CT NIfTI volume:
      - center coronal slice
      - center axial slice
    and concatenate them side-by-side.

    Notes:
    - This utility is provided for convenience. Your own preprocessing pipeline may differ.
    - We do not ship any medical images in this repository.
    """
    import nibabel as nib

    p = Path(nii_path)
    if not p.exists():
        raise FileNotFoundError(f"NIfTI not found: {p}")

    vol = nib.load(str(p)).get_fdata().astype(np.float32)  # (X,Y,Z) or similar
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    x, y, z = vol.shape
    cor = vol[:, y // 2, :]
    axi = vol[:, :, z // 2]

    cor_u8 = _to_uint8(cor, clip_percentiles)
    axi_u8 = _to_uint8(axi, clip_percentiles)

    cor_img = Image.fromarray(cor_u8).convert("L")
    axi_img = Image.fromarray(axi_u8).convert("L")

    # resize while preserving aspect ratio
    def _resize(im: Image.Image) -> Image.Image:
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        return im

    cor_img = _resize(cor_img)
    axi_img = _resize(axi_img)

    # concat horizontally, pad to same height
    h = max(cor_img.size[1], axi_img.size[1])
    w = cor_img.size[0] + axi_img.size[0]
    out = Image.new("L", (w, h), color=0)
    out.paste(cor_img, (0, (h - cor_img.size[1]) // 2))
    out.paste(axi_img, (cor_img.size[0], (h - axi_img.size[1]) // 2))

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path))
    return str(out_path)
