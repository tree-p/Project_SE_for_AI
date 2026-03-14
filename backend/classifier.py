import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import base64
import io
import numpy as np

def preprocess_image(img_rgb: np.ndarray) -> np.ndarray:
    out = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    return out

CLASSES = ["Fresh", "Half-Fresh", "Spoiled"]

TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model(weights_path: str) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    model.eval()
    return model

def classify_from_base64(model: nn.Module, roi_b64: str) -> dict:
    img_bytes = base64.b64decode(roi_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return _classify(model, img)

def classify_from_bytes(model: nn.Module, image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _classify(model, img)

def _classify(model: nn.Module, img: Image.Image) -> dict:
    img_np = np.array(img)
    img_np = preprocess_image(img_np)
    img = Image.fromarray(img_np)

    tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    predicted_class = CLASSES[pred_idx]

    if predicted_class == "Fresh":
        freshness = "สด"
        advice = "ก่อนเก็บเนื้อวัวควรล้าง..."
    else:
        freshness = "ไม่สด"
        advice = "ไม่ควรบริโภค ควรทิ้งทันที"

    return {
        "predicted_class": predicted_class,
        "freshness": freshness,
        "advice": advice,
        "probabilities": {
            cls: round(float(p), 6)
            for cls, p in zip(CLASSES, probs)
        }
    }