"""Implementation of the Harris corner detector.

Coordinate convention:
- Images: img[row, col] or img[y, x]
- Keypoints: stored as [x, y] where x=col, y=row
- Gradients: Ix = horizontal (along x/columns), Iy = vertical (along y/rows)
"""

from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage  # for the scipy.ndimage.maximum_filter
from scipy import signal  # for the scipy.signal.convolve2d function


# Harris corner detector
def extract_harris(
    img: np.ndarray, sigma: float = 1.0, k: float = 0.05, thresh: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Harris corners from an image.

    Args:
        img (np.ndarray): Gray-scale input image of shape (H, W).
        sigma (float, optional): Gaussian kernel standard deviation. Suggested range [0.5, 2.0].
        k (float, optional): Harris detector free parameter. Suggested range [0.04, 0.06].
        thresh (float, optional): Threshold for corner strength. Suggested range [1e-6, 1e-4].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Detected corners and their strength.
    """
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # Note: In numpy arrays, img[row, col] = img[y, x]
    # So Ix is the gradient along columns (x), Iy along rows (y)
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    print("Hello World!")
    raise NotImplementedError("Implement the image gradients computation.")

    # 2. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum
    raise NotImplementedError("Implement the auto-correlation matrix computation.")

    # 3. Compute Harris response function C
    # TODO: compute the Harris response function C here
    raise NotImplementedError("Implement the Harris response function computation.")

    # 4. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition;
    # Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    raise NotImplementedError("Implement the corner detection and non-maximum suppression.")

    return corners, C
