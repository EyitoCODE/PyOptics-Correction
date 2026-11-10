"""Forward Brown-Conrady distortion with known synthetic camera parameters."""
import cv2
import numpy as np


class OpticalProcessor:
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        self.width, self.height = width, height
        self.camera_matrix = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
        # OpenCV forward projection: negative k1 gives barrel distortion.
        self.dist_coeffs_barrel = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float64)
        self.dist_coeffs_pincushion = np.array([0.3, -0.1, 0, 0, 0], dtype=np.float64)
        self._maps = {}

    def coefficients(self, distortion_type):
        if distortion_type == "barrel":
            return self.dist_coeffs_barrel
        if distortion_type == "pincushion":
            return self.dist_coeffs_pincushion
        raise ValueError("distortion_type must be barrel or pincushion")

    def apply_distortion(self, frame, distortion_type="barrel"):
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame size must match the synthetic camera")
        if distortion_type not in self._maps:
            # Destination pixels are distorted; sample their ideal source positions.
            yy, xx = np.indices((self.height, self.width), dtype=np.float32)
            pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
            ideal = cv2.undistortPointsIter(
                pixels, self.camera_matrix, self.coefficients(distortion_type),
                None, self.camera_matrix,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9),
            ).reshape(self.height, self.width, 2)
            self._maps[distortion_type] = (
                ideal[:, :, 0].astype(np.float32), ideal[:, :, 1].astype(np.float32))
        return cv2.remap(frame, *self._maps[distortion_type], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def apply_correction(self, frame, distortion_type="barrel"):
        return cv2.undistort(frame, self.camera_matrix,
                             self.coefficients(distortion_type), None, self.camera_matrix)
