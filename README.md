# PyOptics: Real-Time Lens Distortion & Correction Engine

![PyOptics Demo](assets/demo_gif.gif) 

###  Engineering Challenge
Commercial optical sensors and camera lenses often suffer from radial distortion, degrading signal and image integrity. Instead of relying on expensive, perfectly ground optical hardware, modern systems use digital signal processing to correct these aberrations. 

This project is a real-time computer vision pipeline engineered to simulate physical optical defects and algorithmically rectify them, bridging the gap between **Optical Physics** and **Software Engineering**.

###  Technology Stack & Architecture
* **Language:** Python 3.9+
* **Computer Vision:** OpenCV (`cv2`)
* **Matrix Operations:** NumPy

The codebase strictly adheres to the **Separation of Concerns** principle:
* `src/distortion_model.py`: Acts as the physics engine, injecting mathematical radial distortion (Barrel/Pincushion) into a raw video feed.
* `src/calibration.py`: Acts as the DSP (Digital Signal Processor), computing the optimal camera matrix and rectifying the warped image back to a flat optical field.

###  The Optical Physics (Brown's Model)
Real-world lenses bend light more at the edges than at the optical center. This project models that physical reality using the radial distortion polynomial:

$$r_{distorted} = r_{ideal} (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$



* **If $k_1 > 0$:** Produces **Barrel Distortion** (Common in wide-angle/fisheye lenses). Straight lines bow outward.
* **If $k_1 < 0$:** Produces **Pincushion Distortion** (Common in telephoto lenses). Straight lines bow inward.

###  How to Run the Pipeline
1. Clone the repository to your local machine.
2. Ensure you have a webcam connected.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


###  System Controls
While the video feed is active, use the following keystrokes to interact with the engine:

b : Inject Barrel Distortion (Simulates a fisheye lens defect)

p : Inject Pincushion Distortion

c : Apply Algorithmic Correction (Rectifies the distortion in real-time)

n : Return to normal raw feed

q : Terminate stream
