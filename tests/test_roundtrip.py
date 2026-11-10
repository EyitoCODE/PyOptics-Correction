"""Numerical regression tests for forward distortion and inverse correction."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distortion_model import OpticalProcessor
from calibration import LensCalibrator
from main import process_frame


class RoundTripTests(unittest.TestCase):
    def test_both_distortions_reduce_grid_reconstruction_error(self):
        image = np.full((360, 480, 3), 220, dtype=np.uint8)
        for x in range(20, 480, 30):
            cv2.line(image, (x, 0), (x, 359), (30, 30, 30), 2)
        for y in range(20, 360, 30):
            cv2.line(image, (0, y), (479, y), (30, 30, 30), 2)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        processor = OpticalProcessor(480, 360)
        crop = np.s_[55:-55, 70:-70]
        for kind in ("barrel", "pincushion"):
            with self.subTest(kind=kind):
                calibrator = LensCalibrator(processor.camera_matrix, processor.coefficients(kind))
                distorted, corrected = process_frame(image, processor, calibrator, kind)
                before = np.mean((distorted[crop].astype(float) - image[crop]) ** 2)
                after = np.mean((corrected[crop].astype(float) - image[crop]) ** 2)
                self.assertGreater(before, 100)
                self.assertLess(after, before * 0.2)
                self.assertEqual(corrected.shape, image.shape)

    def test_unknown_distortion_is_rejected(self):
        with self.assertRaises(ValueError):
            OpticalProcessor(10, 10).coefficients("unknown")


if __name__ == "__main__":
    unittest.main()
