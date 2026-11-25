import cv2
import numpy as np


def estimate_affine(source, target, reliability, threshold=3., min_reliability=.5):
    source, target, reliability = map(np.asarray, (source, target, reliability))
    valid = np.isfinite(source).all(1) & np.isfinite(target).all(1) & (reliability >= min_reliability)
    if valid.sum() < 3 or np.linalg.matrix_rank(source[valid] - source[valid].mean(0)) < 2:
        return None, np.zeros(len(source), dtype=bool)
    matrix, inliers = cv2.estimateAffine2D(source[valid].astype(np.float32), target[valid].astype(np.float32),
                                         method=cv2.RANSAC, ransacReprojThreshold=threshold,
                                         maxIters=5000, confidence=.999, refineIters=10)
    selected = np.zeros(len(source), dtype=bool)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        return None, selected
    selected[np.where(valid)[0]] = inliers.ravel().astype(bool)
    return matrix, selected
