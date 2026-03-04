import cv2
import numpy as np

class LensCalibrator:
    """
    Handles the digital signal processing to correct radial distortion.
    
    In a real-world scenario (like Ciena or Viavi), you would use a checkerboard 
    pattern to calculate the exact camera matrix and distortion coefficients 
    of a physical lens. Here, we use known coefficients to reverse the math.
    """
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def rectify_image(self, frame):
        """
        Applies algorithmic correction matrices to flatten the optical field.
        """
        h, w = frame.shape[:2]
        
        # Calculate the optimal camera matrix based on the free scaling parameter.
        # This prevents black borders from appearing after the image is un-warped.
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort the image (The actual rectification step)
        rectified_frame = cv2.undistort(
            frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )
        
        # Crop the image using the Region of Interest (ROI) to remove the curved edges
        x, y, w_roi, h_roi = roi
        rectified_frame = rectified_frame[y:y+h_roi, x:x+w_roi]
        
        # Resize back to original dimensions for consistent video output
        rectified_frame = cv2.resize(rectified_frame, (w, h))
        
        return rectified_frame