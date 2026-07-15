# Thiago

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-111F68)](https://docs.ultralytics.com/models/yolov8/)
[![Repository checks](https://github.com/GioEspinoza/thiago/actions/workflows/ci.yml/badge.svg)](https://github.com/GioEspinoza/thiago/actions/workflows/ci.yml)

Thiago is an early-stage computer-vision perception system for a planned
assistive mobile robot. The current prototype detects and tracks people from a
live camera, selects the tallest visible tracked person, and extracts OSNet
appearance features for future person re-identification work.

The long-term goal is a robotic platform that can follow a designated child and
support caregiver-facing monitoring. This repository currently covers only the
software perception prototype—not autonomous robotics, identity recognition,
behavioral analysis, or caregiver notifications.

## Current capabilities

- Capture a live webcam feed with OpenCV
- Detect people using YOLOv8
- Track multiple people across frames with BoT-SORT
- Filter low-confidence or untracked detections
- Select the tallest visible tracked person as the temporary target
- Crop each detected person and extract OSNet appearance features
- Visualize target and non-target detections in real time

## Pipeline

```text
Camera frame
    ↓
YOLOv8 person detection
    ↓
BoT-SORT track IDs
    ↓
Bounding-box filtering and person crops
    ↓
Tallest-person target selection
    ↓
OSNet feature extraction
    ↓
Annotated live preview
```

The extracted OSNet features are not yet stored or compared. Persistent
identity matching remains roadmap work.

## Local setup

### Requirements

- Python 3.12
- A working webcam
- CPU execution or a CUDA-capable environment supported by PyTorch
- YOLOv8 Nano weights
- An OSNet x1.0 ImageNet checkpoint

### Installation

```bash
git clone https://github.com/GioEspinoza/thiago.git
cd thiago
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```powershell
venv\Scripts\activate
```

Place the model files at the repository root using the names expected by the
prototype:

```text
yolov8n.pt
osnet_x1_0_imagenet.pth
```

Model weights are intentionally excluded from Git because they are large
third-party artifacts. Ultralytics may download YOLO weights automatically;
the OSNet checkpoint must be available at the configured local path.

### Run

```bash
python testcam.py
```

Press `q` in the camera window to exit.

## Technology

- Python
- OpenCV
- Ultralytics YOLOv8
- BoT-SORT
- PyTorch
- Torchreid
- OSNet

## Project status

| Phase | Status |
|---|---|
| Webcam capture and person detection | Implemented |
| Multi-object tracking | Implemented |
| Temporary tallest-person selection | Implemented |
| OSNet feature extraction | Implemented |
| Reference identity enrollment | Planned |
| Persistent identity matching | Planned |
| Behavioral event detection | Planned |
| Caregiver notifications | Planned |
| Motor and sensor integration | Planned |

## Privacy and safety

This prototype processes live camera frames and is not production-ready.
Obtain consent before recording or monitoring anyone, especially children.
Do not use the current tallest-person heuristic as an identity or safety
decision system.

## Authors

- Software: [Giovanni Espinoza](https://github.com/GioEspinoza)
- Hardware collaboration: Giselle Espinoza
