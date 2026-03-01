from fastapi import FastAPI, UploadFile, File, HTTPException
from localize import localize_beef

app = FastAPI(title="ROI Localization Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "roi_localization"}

@app.post("/localize")
async def localize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        result = localize_beef(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result