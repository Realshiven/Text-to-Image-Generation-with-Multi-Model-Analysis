from fastapi import FastAPI, UploadFile, File
from PIL import Image
import uuid

from app.diffusion import generate_image
from app.clip_model import analyze_image
from app.sam_model import segment_image
from app.utils import image_to_base64

app = FastAPI()

@app.post("/generate")
def generate(prompt: str):
    image = generate_image(prompt)
    clip_result = analyze_image(image)
    masks = segment_image(image)

    return {
        "request_id": str(uuid.uuid4()),
        "generated_image": image_to_base64(image),
        "clip_analysis": clip_result,
        "basic_segmentation": {
            "num_masks": len(masks)
        }
    }

@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    clip_result = analyze_image(image)
    masks = segment_image(image)

    return {
        "request_id": str(uuid.uuid4()),
        "clip_analysis": clip_result,
        "basic_segmentation": {
            "num_masks": len(masks)
        }
    }
