import os
import av
import torch
import numpy as np
import logging
from tqdm import tqdm
from .transnetv2_pytorch import TransNetV2

def init_model(device: torch.device | None = None):
    # Init logger
    logger = logging.getLogger(__name__)
    
    # Set device
    if (device is None) and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Init model
    model = TransNetV2()
    state_dict = torch.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/transnetv2-pytorch-weights.pth"
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
     
    # If CUDA - optimize
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")   
               
    # Return model
    return {
        "LOGGER": logger,
        "MODEL": model,
        "DEVICE": device,
    }
        
def extract_frames(
        video_path: str,
        logger,
        target_height: int = 27,
        target_width: int = 48,
        show_progressbar: bool = False
):
    """
    Extracts frames from a video with progress tracking.
    """
    logger.info(f"Opening video: {video_path}") 
    frames = []
    try:
        # Try to open video
        with av.open(video_path) as container:
            # If can not open
            if not container.streams.video:
                raise ValueError(f"No video stream found: {video_path}")
            
            # Make stream
            stream = container.streams.video[0]
            stream.thread_type = "AUTO" 
            
            # Make decoder
            total_frames = stream.frames or None
            it = container.decode(video=0)
            
            # Informing
            if show_progressbar:
                it = tqdm(it, total=total_frames, desc="Extracting frames", unit="frame")
            
            # Decode frames 
            for frame in it:
                # Resize + rgb24
                frame = frame.reformat(
                    width=target_width,
                    height=target_height,
                    format="rgb24",
                )

                arr = frame.to_ndarray()
                frames.append(arr)
         
    except (av.FFmpegError, OSError, ValueError) as e:
        logger.error(f"Failed to open/decode video: {video_path}. PyAV error: {e}")
        raise
    
    logger.info(f"Extracted {len(frames)} frames")
    return np.asarray(frames, dtype=np.uint8)    

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

def predict_raw(model, video, device):
    """
    Performs inference on the video using the TransNetV2 MODEL.
    """  
    with torch.inference_mode():
        predictions = []
        for inp in input_iterator(video):
            video_tensor = torch.from_numpy(inp).to(device)
            single_frame_pred, all_frame_pred = model(video_tensor)
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
            predictions.append((single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0]))
        single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
        return video.shape[0], single_frame_pred

def predict_video(video_path: str, model_container, show_progressbar: bool = False):
    """
    Detects shot boundaries in a video file using the TransNetV2 MODEL.
    """
    # Get settings
    logger = model_container["LOGGER"]
    model = model_container["MODEL"]
    device = model_container["DEVICE"]
    
    # Extract frames
    frames = extract_frames(video_path, logger, show_progressbar=show_progressbar)
    
    # Predict
    _, single_frame_pred = predict_raw(model, frames, device=device)
    
    # Get scenes
    scenes = predictions_to_scenes(single_frame_pred)
    logger.info(f"Detected {len(scenes)} scenes")
    return scenes