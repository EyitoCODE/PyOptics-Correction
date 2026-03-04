import numpy as np
import cv2

class OpticalProcessor:
    """
    Handles the mathematical simulation of lens distortion and correction.
    
    Physics Concept:
    Ideal lenses project straight lines as straight. Real lenses suffer from 
    'Radial Distortion' where light bends more near the edges of the lens.
    
    Formula (Brown's Model):
    r_distorted = r_ideal * (1 + k1 * r^2 + k2 * r^4 + ...)
    where 'k' coefficients determine the strength of the distortion.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Camera Matrix (Intrinsics): Defines how the camera 'sees' the world.
        # [ fx   0  cx ]
        # [  0  fy  cy ]
        # [  0   0   1 ]
        # fx, fy = Focal Length (Zoom level)
        # cx, cy = Principal Point (Optical Center)
        self.camera_matrix = np.array([
            [width, 0, width / 2],
            [0, width, height / 2],
            [0, 0, 1]
        ], dtype="double")

        # Distortion Coefficients (k1, k2, p1, p2, k3)
        # k1 > 0 : Barrel Distortion (Fish-eye effect) - Lines bow out
        # k1 < 0 : Pincushion Distortion - Lines bow in
        self.dist_coeffs_barrel = np.array([-0.3, 0.1, 0, 0, 0], dtype="double") # Simulates a wide-angle lens
        self.dist_coeffs_pincushion = np.array([0.3, -0.1, 0, 0, 0], dtype="double") # Simulates a telephoto lens error

    def apply_distortion(self, frame, distortion_type='barrel'):
        """
        Simulates a hardware defect by mathematically warping the image.
        Uses cv2.undistort logic in reverse or remap to simulate the error.
        Note: OpenCV doesn't have a direct 'add_distortion' function, so we 
        manipulate the camera matrix to simulate the visual effect.
        """
        
        # Select the coefficients based on the desired physical defect
        if distortion_type == 'barrel':
            k = self.dist_coeffs_barrel
        else:
            k = self.dist_coeffs_pincushion

        # Get the optimal new camera matrix based on the free scaling parameter 'alpha=1'
        # This calculates a valid region of interest (ROI) for the warped image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, k, (self.width, self.height), 1, (self.width, self.height)
        )

        # Standard correction pipeline (re-purposed here to show the effect of 'k' values)
        # In a real scenario, this 'undistort' removes error. 
        # By feeding it extreme 'k' values, we visualize what high distortion looks like.
        result = cv2.undistort(frame, self.camera_matrix, k, None, new_camera_matrix)
        
        return result

    def apply_correction(self, frame):
        """
        The Solution: Applies digital signal processing to flatten the field 
        and correct the straight lines.
        """
        # We assume the 'barrel' coefficients are the error we want to fix.
        # To fix it, we use the same coefficients. OpenCV's undistort inverse maps 
        # the distorted pixels back to their ideal positions.
        
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs_barrel, (w,h), 1, (w,h)
        )
        
        # un-distort
        result = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs_barrel, None, new_camera_matrix)
        
        # Crop the image to remove the black curved borders caused by warping
        x, y, w, h = roi
        result = result[y:y+h, x:x+w]
        
        return result