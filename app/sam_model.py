import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).parent / "sam_vit_b.pth"

sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
sam.eval()

predictor = SamPredictor(sam)

def segment_image(image: Image.Image):
    image_np = np.array(image)
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    masks, _, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),
        multimask_output=False
    )

    return masks.tolist()
