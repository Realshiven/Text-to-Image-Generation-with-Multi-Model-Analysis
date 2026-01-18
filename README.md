# ML Assignment – Stable Diffusion + CLIP + SAM (FastAPI)

This project implements a FastAPI service that:

- Generates images from text prompts using **Stable Diffusion**
- Analyzes generated or uploaded images using **CLIP**
- Performs basic image segmentation using **Segment Anything (SAM)**

---

## 🚀 Features

- **POST /generate**
  - Input: text prompt
  - Output:
    - Base64-encoded generated image
    - CLIP-based image-text analysis
    - Basic segmentation metadata (number of masks)

- **POST /analyze**
  - Input: uploaded image
  - Output:
    - CLIP analysis
    - Basic segmentation metadata

---

## 🧠 Models Used

- Stable Diffusion (via `diffusers`)
- CLIP (OpenAI)
- Segment Anything Model (ViT-B)

---

## 🗂 Project Structure


ml_assignment/
├── app/
│ ├── main.py
│ ├── diffusion.py
│ ├── clip_model.py
│ ├── sam_model.py
│ ├── utils.py
│ └── sam_vit_b.pth
├── tests/
│ └── test_api.py
├── generated_output.png
├── save_image.py
├── requirements.txt
└── README.md
