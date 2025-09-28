"""Visualization utilities for keypoints and matches.

Coordinate convention:
- Images: img[row, col] or img[y, x]
- Keypoints: stored as [x, y] where x=col, y=row
- Gradients: Ix = horizontal (along x/columns), Iy = vertical (along y/rows)
"""

import copy
from typing import Tuple

import cv2
import numpy as np


def draw_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw keypoints on the image.

    Args:
        img (np.ndarray): Input image.
        keypoints (np.ndarray): Keypoint locations of shape (q, 2).
        color (Tuple[int, int, int], optional): Circle color. Defaults to (0, 0, 255).
        thickness (int, optional): Circle thickness. Defaults to 2.

    Returns:
        np.ndarray: Image with drawn keypoints.
    """
    if len(img.shape) == 2:
        img = img[:, :, None].repeat(3, 2)
    if keypoints is None:
        raise ValueError("Error! Keypoints should not be None")
    keypoints = np.array(keypoints)
    for p in keypoints.tolist():
        pos_x, pos_y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (pos_x, pos_y), thickness, color, -1)
    return img


def draw_segments(
    img: np.ndarray,
    segments: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw line segments on the image.
    Args:
        img (np.ndarray): Input image.
        segments (np.ndarray): Line segments containing endpoints of shape (m, 4).
        color (Tuple[int, int, int], optional): Line color. Defaults to (255, 0, 0).
        thickness (int, optional): Line thickness. Defaults to 2.

    Returns:
        np.ndarray: Image with drawn line segments.
    """
    for s in segments:
        p1 = (int(round(s[0])), int(round(s[1])))
        p2 = (int(round(s[2])), int(round(s[3])))
        cv2.line(img, p1, p2, color, thickness)
    return img


def plot_image_with_keypoints(
    fname_out: str,
    img: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
):
    """Plot keypoints on the image and save the result.

    Args:
        fname_out (str): Output filename.
        img (np.ndarray): Input image.
        keypoints (np.ndarray): Keypoint locations of shape (q, 2).
        color (Tuple[int, int, int], optional): Circle color. Defaults to (0, 0, 255).
        thickness (int, optional): Circle thickness. Defaults to 2.
    """
    img = copy.deepcopy(img)
    img_keypoints = draw_keypoints(img, keypoints, color=color, thickness=thickness)
    cv2.imwrite(fname_out, img_keypoints)
    print(
        "[LOG] Number of keypoints: {0}. Writing keypoints visualization to {1}".format(
            keypoints.shape[0], fname_out
        )
    )


def plot_image_pair_with_matches(
    fname_out: str,
    img1: np.ndarray,
    keypoints1: np.ndarray,
    img2: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
):
    """Plot matches between two images and save the result.

    Args:
        fname_out (str): Output filename.
        img1 (np.ndarray): First input image.
        keypoints1 (np.ndarray): Keypoint locations in the first image.
        img2 (np.ndarray): Second input image.
        keypoints2 (np.ndarray): Keypoint locations in the second image.
        matches (np.ndarray): Matched keypoints between the two images of shape (m, 2).
    """
    # construct full image
    assert img1.shape[0] == img2.shape[0]
    assert img1.shape[1] == img2.shape[1]
    h, w = img1.shape[0], img1.shape[1]
    img = np.concatenate([img1, img2], 1)
    img = img[:, :, None].repeat(3, 2)
    img = draw_keypoints(img, keypoints1, color=(0, 0, 255), thickness=2)
    img = draw_keypoints(
        img, keypoints2 + np.array([w, 0])[None, :], color=(0, 0, 255), thickness=2
    )
    segments = []
    segments.append(keypoints1[matches[:, 0]])
    segments.append(keypoints2[matches[:, 1]] + np.array([w, 0])[None, :])
    segments = np.concatenate(segments, axis=1)
    img = draw_segments(img, segments, color=(255, 0, 0), thickness=1)
    cv2.imwrite(fname_out, img)
    print(
        "[LOG] Number of matches: {0}. Writing matches visualization to {1}".format(
            matches.shape[0], fname_out
        )
    )
