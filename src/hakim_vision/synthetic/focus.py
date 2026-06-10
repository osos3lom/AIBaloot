"""Focus / blur detection.

Ported from `varianceOfLaplacian` in the legacy notebook. Approach attributed to
Adrian Rosebrock, https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def variance_of_laplacian(image: NDArray[np.uint8]) -> float:
    """Return the variance of the Laplacian of `image` as a focus measure.

    Higher values indicate a sharper image; lower values indicate blur. A common
    cutoff for "in focus" is around 100-150 for typical webcam frames, but the
    right threshold is dataset-dependent. The legacy notebook used 120.

    Args:
        image: A 2-D grayscale or 3-D BGR image as a NumPy array.

    Returns:
        Scalar focus measure (non-negative float).

    Raises:
        ValueError: if `image` is not a 2-D or 3-D NumPy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError(
            f"image must be a 2-D or 3-D ndarray, got ndim={getattr(image, 'ndim', None)}"
        )
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


__all__ = ["variance_of_laplacian"]
