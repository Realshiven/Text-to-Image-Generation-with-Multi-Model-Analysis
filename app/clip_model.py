import torch
import clip
from PIL import Image

_model = None
_preprocess = None

def get_clip_model():
    global _model, _preprocess
    if _model is None:
        print("Loading CLIP model lazily...")
        _model, _preprocess = clip.load("ViT-B/32", device="cpu")
    return _model, _preprocess


def analyze_image(image: Image.Image):
    """
    Analyze an image using CLIP.
    Loads model lazily to avoid startup failures.
    """
    model, preprocess = get_clip_model()

    image_tensor = preprocess(image).unsqueeze(0)
    text = clip.tokenize(["a photo", "an image", "a picture"])

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)

    similarity = (image_features @ text_features.T).softmax(dim=-1)

    return {
        "similarity_scores": similarity.tolist()
    }
