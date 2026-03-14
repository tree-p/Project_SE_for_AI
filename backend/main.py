from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


from beef_localization import localize_beef
from classifier import load_model, classify_from_base64
from confidence import calculate_confidence

app = FastAPI()


origins = [
    "http://192.168.1.40:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse

@app.options("/analyze")
async def options_analyze():
    return JSONResponse(content={"message": "OK"})

# โหลดโมเดลครั้งเดียวตอน start server
model = load_model(r"C:\Users\deamk\Downloads\BACKEND\backend\models\beef_model.pt")


@app.post("/analyze")
async def analyze_beef(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # -------------------------
    # 1️⃣ Localization (รวม Quality + Normalize)
    # -------------------------

    roi_result = localize_beef(image_bytes)

    # ถ้าไม่ผ่าน quality gate
    if roi_result.get("status") == "rejected":
        return roi_result

    roi_b64 = roi_result["roi_base64"]

    # -------------------------
    # 2️⃣ Classification
    # -------------------------

    result = classify_from_base64(model, roi_b64)

    # -------------------------
    # 3️⃣ Confidence
    # -------------------------

    confidence = calculate_confidence(result["probabilities"])

    # -------------------------
    # Final Response
    # -------------------------

    return {
        "prediction": result["predicted_class"],
        "freshness": result["freshness"],
        "advice": result["advice"],

        "probabilities": result["probabilities"],
        "confidence": confidence,

        "roi_bbox": roi_result["bbox"],
        "localization_confidence": roi_result["localization_confidence"],
        "fallback": roi_result["fallback"]
    }