## TransNetV2 Fork with CUDA, PyTorch, Logging, and Progress Bar

This repository is a fork of [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) that introduces several enhancements:

* **OpenCV CUDA Integration**: Accelerated frame extraction leveraging NVIDIA CUDA.
* **PyTorch Backend**: Native PyTorch model loading and inference for easier integration and extension.
* **Logging**: Structured logging via Python `logging` module for better observability.
* **Progress Bar**: Real-time progress feedback using `tqdm` during frame extraction and scene detection.

---

### Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Configuration](#configuration)
6. [Contributing](#contributing)
7. [License](#license)

---

## Features

* **Fast Frame Extraction**: Utilizes OpenCV with CUDA support, if available, falling back to CPU otherwise.
* **PyTorch Model**: Loads TransNetV2 weights via PyTorch, supports GPU/CPU inference.
* **Scene Boundary Detection**: Produces accurate shot boundary detection with configurable thresholds.
* **Logging & Monitoring**: INFO-level logs for key steps and errors.
* **Progress Feedback**: `tqdm` progress bars for both frame extraction and scene prediction.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/TransNetV2.git
   ```
3. Install dependencies (tested on Python 3.12):

   ```bash
   pip install -r requirements.txt
   ```
3. Install PyTorch:
   
   https://pytorch.org/get-started/locally/
4. Build and Install OpenCV CUDA

   https://machinelearningprojects.net/build-opencv-with-cuda-and-cudnn/
5. Install package:
   ```bash
   python setup.py install
   ``` 

## Usage

```python
from transnetv2pt import predict_video

# Detect scenes in a video file
target = "path/to/video.mp4"
scenes = predict_video(str(target), device='cuda', show_progressbar=True)
print(scenes)
```

## Examples

#### Extract Key Frames

```python
from pathlib import Path
from transnetv2pt import predict_video
import cv2

video_path = Path("video.mkv")
scenes = predict_video(str(video_path), show_progressbar=True)

cap = cv2.VideoCapture(str(video_path))
for i, (start, end) in enumerate(scenes):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"scene_{i}_start.png", frame)
cap.release()
```

## Configuration

* **CUDA Support**: Automatic detection via `cv2.cuda.getCudaEnabledDeviceCount()`.
* **Device Selection**: Pass `device='cpu'` or `device='cuda'` to `predict_video()`.
* **Progress Bars**: Enable via `show_progressbar=True` in both frame extraction and scene detection.

## Contributing

Contributions are welcome! Please open issues and submit pull requests for bug fixes and enhancements.

## License

This project inherits the MIT License from the original TransNetV2 repository. See `LICENSE` for details.
