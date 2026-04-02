from typing import List, Optional



from utils.config import get_blendshape_names





def make_audio_chunk_msg(

    chunk_index: int,

    sentence_index: int,

    audio_base64: str,

    start_time: float,

    end_time: float,

    sample_rate: int,

    audio_format: str = "pcm_s16le",

    channels: int = 1,

    audio_bytes_base64: Optional[str] = None,

    is_final: bool = False,

) -> dict:

    msg = {

        "type": "audio_chunk",

        "chunk_index": chunk_index,

        "sentence_index": sentence_index,

        "audio_base64": audio_base64,

        "start_time": round(start_time, 4),

        "end_time": round(end_time, 4),

        "sample_rate": sample_rate,

        "audio_format": audio_format,

        "channels": channels,

        "is_final": is_final,

    }



    if audio_bytes_base64 is not None:

        msg["audio_bytes_base64"] = audio_bytes_base64



    return msg





def _raw_frames_to_named(

    raw_frames: List[List[float]],

    chunk_start_time: float,

    frame_rate: int,

) -> List[dict]:

    """Convert raw frame arrays into per-frame dicts with names + timestamps."""

    names = get_blendshape_names()

    result = []

    for i, frame in enumerate(raw_frames):

        result.append({

            "timestamp": round(chunk_start_time + i / frame_rate, 6),

            "blendshapes": {

                names[j]: round(float(v), 6) for j, v in enumerate(frame)

            },

        })

    return result





def make_blendshapes_msg(

    chunk_index: int,

    sentence_index: int,

    frames: List[List[float]],

    start_time: float,

    end_time: float,

    frame_rate: int,

    is_final: bool = False,

) -> dict:

    return {

        "type": "blendshapes",

        "chunk_index": chunk_index,

        "sentence_index": sentence_index,

        "frames": _raw_frames_to_named(frames, start_time, frame_rate),

        "start_time": round(start_time, 4),

        "end_time": round(end_time, 4),

        "frame_rate": frame_rate,

        "mapping": get_blendshape_names(),

        "is_final": is_final,

    }





def make_text_chunk_msg(

    sentence_index: int,

    text: str,

    is_final: bool = False,

) -> dict:

    return {

        "type": "text_chunk",

        "sentence_index": sentence_index,

        "text": text,

        "is_final": is_final,

    }





def make_idle_frames_msg(

    frames: List[List[float]],

    start_time: float,

    end_time: float,

    frame_rate: int,

) -> dict:

    return {

        "type": "idle_frames",

        "frames": _raw_frames_to_named(frames, start_time, frame_rate),

        "start_time": round(start_time, 4),

        "end_time": round(end_time, 4),

        "frame_rate": frame_rate,

    }





def make_status_msg(status: str, message: str) -> dict:

    return {

        "type": "status",

        "status": status,

        "message": message,

    }

