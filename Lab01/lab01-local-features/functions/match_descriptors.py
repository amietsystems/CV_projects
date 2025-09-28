"""Implementation of descriptor matching methods.

Coordinate convention:
- Images: img[row, col] or img[y, x]
- Keypoints: stored as [x, y] where x=col, y=row
- Gradients: Ix = horizontal (along x/columns), Iy = vertical (along y/rows)
"""

from typing import Literal

import numpy as np


def ssd(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """Compute the sum of squared differences (SSD) between two descriptor sets.

    Args:
        desc1 (np.ndarray): Descriptor set 1 of shape (q1, feature_dim).
        desc2 (np.ndarray): Descriptor set 2 of shape (q2, feature_dim).

    Returns:
        np.ndarray: SSD distances of shape (q1, q2).
    """
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    raise NotImplementedError("Implement the SSD computation.")


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: Literal["one_way", "mutual", "ratio"] = "one_way",
    ratio_thresh: float = 0.5,
) -> np.ndarray:
    """Match feature descriptors between two sets.

    Args:
        desc1 (np.ndarray): Descriptor set 1 of shape (q1, feature_dim).
        desc2 (np.ndarray): Descriptor set 2 of shape (q2, feature_dim).
        method (Literal["one_way", "mutual", "ratio"], optional): Matching method.
        ratio_thresh (float, optional): Ratio threshold for matching.

    Returns:
        np.ndarray: Array of shape (m x 2) storing the indices of the matches.
    """
    assert (
        desc1.shape[1] == desc2.shape[1]
    ), f"Feature dimensions do not match: {desc1.shape[1]} vs {desc2.shape[1]}"

    # Handle empty descriptor sets
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        print("Warning: One of the descriptor sets is empty. No matches can be found.")
        return np.empty((0, 2), dtype=int)

    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way":  # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        raise NotImplementedError("Implement the one-way nearest neighbor matching (I1 -> I2).")
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        raise NotImplementedError("Implement the mutual nearest neighbor matching (I1 <-> I2).")
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        raise NotImplementedError("Implement the ratio test matching.")
    else:
        raise NotImplementedError(f"Unknown matching method {method}.")
    return matches
