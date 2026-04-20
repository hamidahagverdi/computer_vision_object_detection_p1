# License Plate Detection Project


## Project Flow

1. Use a frontal car image or front-camera style video.
2. Run an improved classical plate-localization pipeline.
3. Save the annotated result and cropped plate.
4. Compare image and video behavior in the report/presentation.

## Included Demo Data

- `data/sample_images/mazda_plate_demo.png`
- `data/raw_videos/demo_dashcam_plate.mp4`

The video is a **demo dashcam-style sample generated from the included image**, so the detector works cleanly out of the box.
You can later replace it with a real dashcam video if your professor wants a more realistic evaluation.

## Folder Structure

```text
anpr_professional_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_images/
│   └── raw_videos/
├── src/
│   └── detect_plate.py
├── results/
│   ├── image_outputs/
│   └── video_outputs/
└── reports/
```

## Installation

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run on the Included Image

```bash
python3 src/detect_plate.py image   --input data/sample_images/mazda_plate_demo.png   --output-dir results/image_outputs
```

This saves:
- annotated image
- plate crop
- debug mask

## Run on the Included Video

```bash
python3 src/detect_plate.py video   --input data/raw_videos/demo_dashcam_plate.mp4   --output results/video_outputs/demo_dashcam_plate_annotated.mp4
```

## What Was Improved Compared to the Noisy Version

The earlier version drew many boxes because it only used a weak width/height rule.
This improved version adds:

- a **region of interest** focused on the likely plate area
- **bright-region extraction** for white plate backgrounds
- stronger **aspect-ratio and area filtering**
- a **scoring function** that prefers plate-like texture and position
- simple **temporal smoothing** for video

## Short Explanation for Presentation

- Classical CV uses hand-crafted rules.
- It is fast and interpretable.
- It can work well in controlled frontal views.
- It becomes less robust in complex real-world scenes.
- ML-based methods are usually stronger when lighting, angle, blur, and background vary a lot.

## Suggested Presentation Claim

> The improved classical pipeline can localize the plate reliably in controlled frontal views, but ML-based detectors are more robust for real driving footage with strong variability.

## Replace With Your Own Real Data

You can place your own files here:
- images: `data/sample_images/`
- videos: `data/raw_videos/`

Then run the same commands with the new paths.


## Refinements in v2
- better plate candidate scoring using both bright-plate and dark-character cues
- temporal smoothing for video to reduce box jumping in early frames
- previous-frame guided search for more stable tracking
- optional `--resize-width` for harder videos

### Better video command
```bash
python3 src/detect_plate.py video --input data/raw_videos/license_plate_detection_countries.mp4 --output results/video_outputs/license_plate_detection_countries_annotated.mp4 --resize-width 960
```


## v3 note
- Image mode is tighter again and crops only the plate region more precisely.
- Video mode keeps the temporal smoothing from v2 for better frame-to-frame stability.


## v4 Improvements
- safer crop padding so edge digits are not clipped
- stronger video candidate filtering with character-component checks
- non-maximum suppression for overlapping candidates
- more stable frame-to-frame box smoothing


## Added ML-Based Method (YOLO)

This project now includes **two runnable methods on the same input**:

1. **Classical CV** (`src/detect_plate.py`)
2. **YOLO-based detector** (`src/detect_plate_yolo.py`)

This lets you compare how rule-based detection and ML-based detection behave on the same image or video.

### Run Classical Method on an Image

```bash
python3 src/detect_plate.py image --input data/sample_images/car1.jpeg --output-dir results/image_outputs
```

### Run YOLO Method on the Same Image

```bash
python3 src/detect_plate_yolo.py image --input data/sample_images/car1.jpeg --output-dir results/yolo_image_outputs
```

### Run Both Methods and Build a Comparison Montage

```bash
python3 src/compare_methods.py --input data/sample_images/car1.jpeg
```

This saves:
- `results/image_outputs/<name>_detected.png`
- `results/yolo_image_outputs/<name>_yolo_detected.png`
- `results/method_comparison.png`

### Run YOLO on Video

```bash
python3 src/detect_plate_yolo.py video   --input data/raw_videos/lpd.mp4   --output results/yolo_video_outputs/lpd_yolo_annotated.mp4   --resize-width 960
```

### Notes on YOLO Setup

- On the **first run**, `detect_plate_yolo.py` downloads a public license-plate YOLO model into `models/`.
- After that, the same local `.pt` file is reused.
- If you already have your own plate detector weights, replace the file in `models/` or pass `--model path/to/your_model.pt`.
