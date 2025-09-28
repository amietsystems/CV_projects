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
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ix = signal.convolve2d(img, x_kernel, mode='same', boundary='symm')
    y_kernel = np.transpose(x_kernel)
    Iy = signal.convolve2d(img, y_kernel, mode='same', boundary='symm')

    # 2. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Smoothed versions
    # Set neighborhood of M via sigma -> (0,0) means that the neighborhood size is calculated from sigma
    # Weight is Gaussian
    Sxx = cv2.GaussianBlur(Ixx, (0, 0), sigma)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigma)

    # 4D M matrix. (H, W, 2, 2); (2,2) is the M matrix at a given pixel
    M = np.zeros((img.shape[0], img.shape[1], 2, 2))
    M[:, :, 0, 0] = Sxx
    M[:, :, 0, 1] = Sxy
    M[:, :, 1, 0] = Sxy
    M[:, :, 1, 1] = Syy

    # 3. Compute Harris response function C
    # TODO: compute the Harris response function C here
    det_M = (M[:, :, 0, 0] * M[:, :, 1, 1]) - (M[:, :, 1, 0] * M[:, :, 0, 1])
    trace_M = M[:, :, 0, 0] + M[:, :, 1, 1]
    C = det_M - k * (trace_M ** 2)

    # 4. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition;
    # Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    c = C.copy()
    local_max = ndimage.maximum_filter(c, size=3, mode="reflect") # maximum harris response in its 3x3 neighborhood
    peak_mask = (c == local_max) & (c > thresh) # get the mask of local peaks above threshold
    # corners [x, y] where x=col, y=row
    ys, xs = np.where(peak_mask)
    corners = np.stack((xs, ys), axis=-1)

    return corners, C
