config = {
    'sr': 88200,
    'frame_rate': 60,
    'hidden_dim': 1024,
    'n_layers': 8,
    'num_heads': 16,
    'dropout': 0.0,
    'output_dim': 68, # if you trained your own, this should also be 61
    'input_dim': 256,
    'frame_size': 128,
    'use_half_precision': False
}

# ---------------------------------------------------------------------------
# ARKit 52 Blendshape Names (Apple standard)
# https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
#
# The model outputs 68 values per frame.  The mapping below assigns the
# standard ARKit names to the first 52 indices and labels indices 52-67 as
# custom extras.  Indices that are zeroed by zero_columns() are marked in
# comments.  Adjust this list to match the actual training data.
# ---------------------------------------------------------------------------

ARKIT_BLENDSHAPE_NAMES = [
    # 0-4  (zeroed by zero_columns)
    "eyeBlinkLeft",           # 0  — zeroed
    "eyeLookDownLeft",        # 1  — zeroed
    "eyeLookInLeft",          # 2  — zeroed
    "eyeLookOutLeft",         # 3  — zeroed
    "eyeLookUpLeft",          # 4  — zeroed
    # 5-6  (active)
    "eyeSquintLeft",          # 5
    "eyeWideLeft",            # 6
    # 7-11 (zeroed by zero_columns)
    "eyeBlinkRight",          # 7  — zeroed
    "eyeLookDownRight",       # 8  — zeroed
    "eyeLookInRight",         # 9  — zeroed
    "eyeLookOutRight",        # 10 — zeroed
    "eyeLookUpRight",         # 11 — zeroed
    # 12-50 (active)
    "eyeSquintRight",         # 12
    "eyeWideRight",           # 13
    "jawForward",             # 14
    "jawLeft",                # 15
    "jawRight",               # 16
    "jawOpen",                # 17
    "mouthClose",             # 18
    "mouthFunnel",            # 19
    "mouthPucker",            # 20
    "mouthLeft",              # 21
    "mouthRight",             # 22
    "mouthSmileLeft",         # 23
    "mouthSmileRight",        # 24
    "mouthFrownLeft",         # 25
    "mouthFrownRight",        # 26
    "mouthDimpleLeft",        # 27
    "mouthDimpleRight",       # 28
    "mouthStretchLeft",       # 29
    "mouthStretchRight",      # 30
    "mouthRollLower",         # 31
    "mouthRollUpper",         # 32
    "mouthShrugLower",        # 33
    "mouthShrugUpper",        # 34
    "mouthPressLeft",         # 35
    "mouthPressRight",        # 36
    "mouthLowerDownLeft",     # 37
    "mouthLowerDownRight",    # 38
    "mouthUpperUpLeft",       # 39
    "mouthUpperUpRight",      # 40
    "browDownLeft",           # 41
    "browDownRight",          # 42
    "browInnerUp",            # 43
    "browOuterUpLeft",        # 44
    "browOuterUpRight",       # 45
    "cheekPuff",              # 46
    "cheekSquintLeft",        # 47
    "cheekSquintRight",       # 48
    "noseSneerLeft",          # 49
    "noseSneerRight",         # 50
    # 51-60 (zeroed by zero_columns)
    "tongueOut",              # 51 — zeroed
    "custom_52",              # 52 — zeroed
    "custom_53",              # 53 — zeroed
    "custom_54",              # 54 — zeroed
    "custom_55",              # 55 — zeroed
    "custom_56",              # 56 — zeroed
    "custom_57",              # 57 — zeroed
    "custom_58",              # 58 — zeroed
    "custom_59",              # 59 — zeroed
    "custom_60",              # 60 — zeroed
    # 61-67 (active — model-specific extras)
    "custom_61",              # 61
    "custom_62",              # 62
    "custom_63",              # 63
    "custom_64",              # 64
    "custom_65",              # 65
    "custom_66",              # 66
    "custom_67",              # 67
]


def get_blendshape_names():
    """Return the blendshape name list (length = output_dim)."""
    return ARKIT_BLENDSHAPE_NAMES


def blendshape_frame_to_dict(frame_values):
    """Convert a raw 68-element array to a {name: value} dict."""
    names = get_blendshape_names()
    return {names[i]: float(v) for i, v in enumerate(frame_values)}


def blendshapes_to_named_frames(frames):
    """
    Convert a list of raw frames (List[List[float]]) to a list of
    dicts with named blendshapes and per-frame timestamps.
    All blendshape values are clamped to [0.0, 1.0] range.
    """
    names = get_blendshape_names()
    frame_rate = config['frame_rate']
    result = []
    for idx, frame in enumerate(frames):
        # Clamp all values to [0.0, 1.0] range for ARKit compatibility
        clamped_values = {
            names[i]: round(max(0.0, min(1.0, float(v))), 6) 
            for i, v in enumerate(frame)
        }
        entry = {
            "timestamp": round(idx / frame_rate, 6),
            "blendshapes": clamped_values,
        }
        result.append(entry)
    return result
