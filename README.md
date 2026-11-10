# PyOptics: Lens Distortion & Correction Demonstrator

A Python/OpenCV prototype showing an original webcam frame, synthetic radial
distortion, and correction of **that same distorted frame** side by side.

## Method

The forward Brown-Conrady radial model is
`r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)`.
The simulator inverse-maps destination pixels to ideal source coordinates; the
rectifier uses matching coefficients. In OpenCV's forward-projection convention,
negative k1 produces barrel distortion and positive k1 produces pincushion
distortion for the coefficients used here. Sampling maps are cached.

## Run

Requires Python 3.10+ and a webcam. From the repository root:

```bash
git clone https://github.com/EyitoCODE/PyOptics-Correction.git
cd PyOptics-Correction
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/main.py --camera 0
```

- **b / p:** select barrel / pincushion distortion.
- **c:** show/hide corrected comparison; **n:** open original view.
- **q:** quit.

## Numerical regression check (no webcam required)

```bash
python -m unittest discover -s tests -v
```

For both distortion types, correction must reduce mean squared reconstruction
error by at least 80% on the central crop of a synthetic grid. Borders are
excluded because missing pixels cannot be recovered.

## Scope and limitations

This is a **known-parameter synthetic demonstration**, not measured physical-camera
calibration. It does not estimate intrinsics from a checkerboard or establish
real-lens accuracy. Interpolation prevents lossless reconstruction; the webcam's
existing lens distortion remains uncalibrated. No hardware-specific frame-rate
claim is made.

Next steps: checkerboard calibration, reprojection-error reporting, and measured
latency with resolution and hardware specified.

- `src/distortion_model.py`: forward rendering and synthetic parameters.
- `src/calibration.py`: cached rectification maps.
- `src/main.py`: frame pipeline and webcam controls.
- `tests/test_roundtrip.py`: numerical regression checks.
