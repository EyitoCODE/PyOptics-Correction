"""Compare a raw frame with its synthetic distortion and correction."""
import argparse
import cv2
from distortion_model import OpticalProcessor
from calibration import LensCalibrator


def process_frame(frame, processor, calibrator, distortion_type):
    distorted = processor.apply_distortion(frame, distortion_type)
    return distorted, calibrator.rectify_image(distorted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a webcam frame; check --camera and permissions")
        height, width = frame.shape[:2]
        processor = OpticalProcessor(width, height)
        calibrators = {kind: LensCalibrator(processor.camera_matrix, processor.coefficients(kind))
                       for kind in ("barrel", "pincushion")}
        kind, show_correction = "barrel", True
        print("b: barrel | p: pincushion | c: toggle correction | n: original | q: quit")
        while ok:
            distorted, corrected = process_frame(frame, processor, calibrators[kind], kind)
            panels = [("Original", frame), (kind.title() + " (synthetic)", distorted)]
            if show_correction:
                panels.append(("Corrected (known parameters)", corrected))
            previews = []
            for label, image in panels:
                preview = cv2.resize(image, (480, max(1, round(height * 480 / width))))
                cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                previews.append(preview)
            cv2.imshow("PyOptics: Original | Distorted | Corrected", cv2.hconcat(previews))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("b"), ord("p")):
                kind = "barrel" if key == ord("b") else "pincushion"
                show_correction = True
            elif key == ord("c"):
                show_correction = not show_correction
            elif key == ord("n"):
                cv2.imshow("PyOptics: Original", frame)
            ok, frame = cap.read()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
