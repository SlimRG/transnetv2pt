import os
import cv2
import torch
import numpy as np
import logging
from tqdm import tqdm
from .transnetv2_pytorch import TransNetV2

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize TransNetV2 model
model = TransNetV2()
state_dict = torch.load(
    f"{os.path.dirname(os.path.abspath(__file__))}/transnetv2-pytorch-weights.pth"
)
model.load_state_dict(state_dict)
model.eval()

def extract_frames_with_opencv(video_path: str, target_height: int = 27, target_width: int = 48, show_progressbar: bool = False):
    """
    Extracts frames from a video using OpenCV with optional CUDA support and progress tracking.
    """
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Initialize progress bar
    progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame") if show_progressbar else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (target_width, target_height))
        frames.append(frame_resized)
        if progress_bar:
            progress_bar.update(1)

    cap.release()
    if progress_bar:
        progress_bar.close()
    logger.info(f"Extracted {len(frames)} frames")
    return np.array(frames)

def input_iterator(frames):
    """
    Generator that yields batches of 100 frames, with padding at the beginning and end.
    """
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start +
        [frames] +
        [end_frame] * no_padded_frames_end, 0
    )

    ptr = 0
    while ptr + 100 <= len(padded_inputs):
        out = padded_inputs[ptr:ptr + 100]
        ptr += 50
        yield out[np.newaxis]

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    """
    Converts model predictions to scene boundaries based on a threshold.
    """
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)

def predict_raw(model, video, device=torch.device('cuda:0')):
    """
    Performs inference on the video using the TransNetV2 model.
    """
    model.to(device)
    with torch.no_grad():
        predictions = []
        for inp in input_iterator(video):
            video_tensor = torch.from_numpy(inp).to(device)
            single_frame_pred, all_frame_pred = model(video_tensor)
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
            predictions.append(
                (single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0]))
        single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
        return video.shape[0], single_frame_pred

def predict_video(video_path: str, device: str = 'cuda', show_progressbar: bool = False):
    """
    Detects shot boundaries in a video file using the TransNetV2 model.
    """
    # Determine device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    frames = extract_frames_with_opencv(video_path, show_progressbar=show_progressbar)
    _, single_frame_pred = predict_raw(model, frames, device=device)
    scenes = predictions_to_scenes(single_frame_pred)
    logger.info(f"Detected {len(scenes)} scenes")
    return scenes