"""Implementation of keypoint filtering and descriptor extraction.

Coordinate convention:
- Images: img[row, col] or img[y, x]
- Keypoints: stored as [x, y] where x=col, y=row
- Gradients: Ix = horizontal (along x/columns), Iy = vertical (along y/rows)
"""

import numpy as np


def filter_keypoints(img: np.ndarray, keypoints: np.ndarray, patch_size: int = 9):
    """Filter keypoints that are too close to the image edges.

    Args:
        img (np.ndarray): Gray-scale image of shape (H, W).
        keypoints (np.ndarray): Keypoint locations of shape (q, 2).
        patch_size (int, optional): Size of the patch to extract. Defaults to 9.

    Returns:
        np.ndarray: Filtered keypoint locations of shape (q', 2).
    """
    # TODO: Filter out keypoints that are too close to the edges
    raise NotImplementedError("Implement the keypoint filtering.")


# The implementation of the patch extraction is already provided here
def extract_patches(img: np.ndarray, keypoints: np.ndarray, patch_size: int = 9) -> np.ndarray:
    """Extract patches from the image at the specified keypoints.

    Args:
        img (np.ndarray): Gray-scale image of shape (H, W).
        keypoints (np.ndarray): Keypoint locations [x, y] of shape (q, 2).
        patch_size (int, optional): Size of the patch to extract. Defaults to 9.

    Returns:
        np.ndarray: Patch descriptors of shape (q, patch_size * patch_size).
    """
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(
        img, ranges[:, None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]
    )  # (q, patch_size, patch_size)
    return desc.reshape(keypoints.shape[0], -1)  # (q, patch_size * patch_size)
