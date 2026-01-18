import torch
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT_PATH = Path(__file__).parent / "sam_vit_b.pth"

_predictor = None  # global cache

def get_predictor():
    global _predictor
    if _predictor is None:
        sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
        sam.eval()
        _predictor = SamPredictor(sam)
    return _predictor

def segment_image(image: Image.Image):
    predictor = get_predictor()

    image_np = np.array(image)
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    masks, _, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),
        multimask_output=False
    )

    return masks.tolist()
