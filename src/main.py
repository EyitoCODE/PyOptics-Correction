import cv2
from distortion_model import OpticalProcessor
from calibration import LensCalibrator  # <--- NEW IMPORT

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Initialize our Optical Physics engine (The Problem)
    processor = OpticalProcessor(width, height)
    
    # Initialize our Calibrator (The Solution)
    # We pass it the 'barrel' coefficients because that is what we are pretending to fix
    calibrator = LensCalibrator(processor.camera_matrix, processor.dist_coeffs_barrel)

    print("--- PyOptics Engine Started ---")
    mode = 'normal'

    while True:
        ret, frame = cap.read()
        if not ret: break

        if mode == 'barrel':
            cv2.putText(frame, "SIMULATION: Barrel Distortion", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = processor.apply_distortion(frame, 'barrel')
            
        elif mode == 'pincushion':
            cv2.putText(frame, "SIMULATION: Pincushion Distortion", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = processor.apply_distortion(frame, 'pincushion')
            
        elif mode == 'correct':
            cv2.putText(frame, "PROCESSING: Algorithmic Correction Applied", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Apply the separated calibration logic
            frame = calibrator.rectify_image(frame)

        cv2.imshow('PyOptics: Real-Time Lens Engine', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('b'): mode = 'barrel'
        elif key == ord('p'): mode = 'pincushion'
        elif key == ord('c'): mode = 'correct'
        elif key == ord('n'): mode = 'normal'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()