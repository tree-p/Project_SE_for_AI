#ก่อนเข้า LOALIZE ของ roi

import cv2
import numpy as np
import base64

# Import ฟังก์ชันจากไฟล์
from quality_gate import image_quality_gate
from normalization import preprocess_color_illumination_tuned

def localize_beef(image_bytes: bytes) -> dict:
    """
    ตรวจจับตำแหน่งเนื้อวัวในภาพด้วย HSV color masking
    Returns: bbox, roi (cropped), confidence
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    h, w = img.shape[:2]
    
    # =========================================================
    # ส่วนเชื่อมต่อ Pipeline การเตรียมข้อมูล
    # 1. แปลงภาพเป็น RGB เพื่อส่งเข้าด่านตรวจ Quality Gate และ Normalization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. ตรวจคุณภาพ (Quality Gate)
    q_result = image_quality_gate(img_rgb)
    if not q_result.passed:
        # ถ้ารูปไม่ผ่านเกณฑ์ ให้คืนค่าสถานะ rejected พร้อมแจ้งเหตุผล
        return {
            "status": "rejected",
            "reasons": q_result.reasons,
            "fallback": True
        }

    # 3. ปรับแสงและสี (Normalization)
    norm_rgb = preprocess_color_illumination_tuned(img_rgb)

    # 4. แปลงกลับเป็น BGR เพื่อให้โค้ดส่วน HSV ด้านล่างทำงานต่อได้ตามปกติ
    img_ready = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
    # =========================================================

    # แปลงเป็น HSV เพื่อตรวจหาสีแดง-ชมพู (เนื้อวัว)
    hsv = cv2.cvtColor(img_ready, cv2.COLOR_BGR2HSV)

    # เนื้อวัวสด: Hue แดง-ชมพู-ส้ม
    mask1 = cv2.inRange(hsv, np.array([0, 40, 50]),   np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # ลบ noise ออก
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: ใช้ทั้งภาพถ้าหาไม่เจอ
        return _build_result(img, 0, 0, w, h, confidence=0.30, fallback=True)

    # largest contours
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < (h * w * 0.05):
        return _build_result(img, 0, 0, w, h, confidence=0.35, fallback=True)

    x, y, bw, bh = cv2.boundingRect(largest)

    # Padding
    pad_x = int(bw * 0.05)
    pad_y = int(bh * 0.05)
    x  = max(0, x - pad_x)
    y  = max(0, y - pad_y)
    bw = min(w - x, bw + pad_x * 2)
    bh = min(h - y, bh + pad_y * 2)

    # Confidence
    coverage   = area / (h * w)
    confidence = min(0.95, 0.50 + coverage * 1.5)

    return _build_result(img, x, y, bw, bh, confidence=confidence)


def _build_result(img, x, y, w, h, confidence: float, fallback: bool = False) -> dict:
    roi = img[y:y+h, x:x+w]

    # Encode ROI to base64
    _, buffer = cv2.imencode('.jpg', roi)
    roi_b64   = base64.b64encode(buffer).decode('utf-8')

    return {
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "roi_base64": roi_b64,
        "localization_confidence": round(confidence, 4),
        "fallback": fallback
    }