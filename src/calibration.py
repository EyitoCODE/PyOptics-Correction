"""Known-parameter rectification; this module does not estimate calibration."""
import cv2


class LensCalibrator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix, self.dist_coeffs = camera_matrix, dist_coeffs
        self._maps = {}

    def rectify_image(self, frame):
        height, width = frame.shape[:2]
        size = (width, height)
        if size not in self._maps:
            self._maps[size] = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                self.camera_matrix, size, cv2.CV_32FC1)
        # Preserve coordinates for meaningful before/after comparisons.
        return cv2.remap(frame, *self._maps[size], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
