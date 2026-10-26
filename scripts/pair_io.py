"""Shared canonical NIfTI pair writing and geometry checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchio as tio


def save_canonical_pair(subject: tio.Subject, image_path: Path, label_path: Path) -> tuple[tio.ScalarImage, tio.LabelMap]:
    """Save one already-canonical subject as an MRA and binary int16 AN pair."""
    source_image = subject["mra"]
    image = tio.ScalarImage(tensor=source_image.data.to(dtype=torch.float32), affine=source_image.affine)
    source_label = subject["label"]
    label = tio.LabelMap(
        tensor=(source_label.data > 0).to(dtype=torch.int16),
        affine=image.affine,
    )
    if tuple(image.orientation) != ("R", "A", "S"):
        raise RuntimeError(f"{subject['case_id']}: non-canonical MRA")
    if image.spatial_shape != source_label.spatial_shape:
        raise RuntimeError(f"{subject['case_id']}: image--label shape mismatch")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(image_path))
    label.save(str(label_path))
    if tuple(label.orientation) != ("R", "A", "S"):
        raise RuntimeError(f"{subject['case_id']}: non-canonical label")
    if image.spatial_shape != label.spatial_shape or not np.allclose(image.affine, label.affine):
        raise RuntimeError(f"{subject['case_id']}: output image--label geometry mismatch")
    return image, label
