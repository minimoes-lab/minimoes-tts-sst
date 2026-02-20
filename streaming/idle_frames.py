import numpy as np

from utils.audio.processing.audio_processing import zero_columns


def generate_idle_frames(
    num_frames: int,
    output_dim: int = 68,
    last_active_frame: np.ndarray = None,
    ease_to_neutral: bool = True,
) -> np.ndarray:
    """
    Generate neutral/idle blendshape frames.

    If *last_active_frame* is provided and *ease_to_neutral* is True,
    smoothly interpolates from that frame to neutral (zeros) over the
    first half of the requested frames.
    """
    frames = np.zeros((num_frames, output_dim), dtype=np.float32)

    if last_active_frame is not None and ease_to_neutral and num_frames > 1:
        ease_count = min(num_frames // 2, 15)  # max ~250ms at 60 fps
        for i in range(ease_count):
            alpha = (i + 1) / (ease_count + 1)
            frames[i] = (1 - alpha) * last_active_frame

    frames = zero_columns(frames)
    return frames
