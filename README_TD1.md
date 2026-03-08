# Tennis Ball Tracking and Motion Analysis

A computer vision project for tracking a tennis ball in video and comparing two analysis pipelines: **YOLO-based object detection** and **SAM-assisted segmentation**. The project extracts ball positions frame by frame, computes relative motion speed, and evaluates the quality of the resulting trajectories using simple quantitative metrics.

## Project Overview

This project investigates how modern vision models can be used for sports object tracking in video. A tennis video is processed frame by frame to detect the ball, estimate its center position, and build a trajectory over time. Two approaches are explored:

- **YOLO detection** using bounding boxes around the tennis ball.
- **YOLO + Segment Anything Model (SAM)** where YOLO provides the ball region and SAM refines the object mask.

The tracked positions are then used to compute normalized speed curves and compare the tracking quality of each method.

## Main Features

- Detects the tennis ball from video frames using **Ultralytics YOLOv8**
- Refines object localization using **Meta's Segment Anything Model (SAM)**
- Extracts ball centroid positions over time
- Computes relative speed from frame-to-frame displacement
- Saves trajectories and speed arrays as NumPy files
- Compares methods using:
  - frame coverage
  - trajectory smoothness
  - speed curve visualization

## Methods

### 1. YOLO-based tracking
The first pipeline uses a pretrained YOLOv8 model to detect objects in each frame. The detection corresponding to the COCO `sports ball` class is selected, and its bounding box center is used as the ball position.

### 2. YOLO + SAM tracking
The second pipeline uses YOLO to localize the ball and then passes the bounding box as a prompt to SAM. The segmentation mask is used to estimate a more refined centroid for the ball.

### 3. Motion analysis
After tracking, the project computes:

- **Frame coverage**: number of frames where the ball was successfully detected
- **Trajectory smoothness**: variance of the second-order trajectory differences
- **Relative speed**: normalized speed curve derived from displacement between consecutive positions

## Project Structure

```text
.
├── TD1.ipynb
├── README.md
└── requirements.txt
```

## Requirements

The project relies on Python packages for computer vision, deep learning, segmentation, and plotting.

Main dependencies:
- `ultralytics`
- `opencv-python`
- `numpy`
- `matplotlib`
- `torch`
- `segment-anything`
- `scipy`

Install them with:

```bash
pip install -r requirements.txt
```

## Input Files

The notebook expects the following external files:

- `tennis.mp4` — input tennis video
- `sam_vit_b_01ec64.pth` — SAM checkpoint file

Place these files in the working directory or update the paths in the notebook accordingly.

## Outputs

Depending on the executed sections, the project generates:

- `yolo_positions.npy`
- `yolo_speeds_rel.npy`
- `sam_positions.npy`
- `sam_speeds_rel.npy`

These files store the extracted trajectories and normalized speed signals for later comparison.

## How to Run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Download the required model/checkpoint files.

3. Place the tennis video in the project directory.

4. Launch Jupyter and open the notebook:

```bash
jupyter notebook TD1.ipynb
```

5. Run the notebook cells in order.

## Notes

- The current implementation uses a pretrained YOLO model (`yolov8s.pt`) and the COCO `sports ball` class.
- For better tennis-ball accuracy, the workflow can be improved with a fine-tuned detector.
- The SAM section requires a downloaded checkpoint and may benefit from GPU acceleration.
- Some notebook installation cells are included for interactive experimentation, but the recommended setup is through `requirements.txt`.

## Future Improvements

- Fine-tune YOLO specifically on tennis-ball data
- Add a Kalman filter or temporal smoothing for more stable trajectories
- Compare additional segmentation or tracking methods
- Convert the notebook into a reusable Python pipeline or package

## GitHub Description

Video-based tennis ball tracking and motion analysis using YOLO and SAM in Python.
