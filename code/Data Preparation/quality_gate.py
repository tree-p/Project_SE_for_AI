import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class QualityResult:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, Any]

def variance_of_laplacian(gray: np.ndarray) -> float:
    """ คำนวณค่าความเบลอของภาพ """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_contrast_lab(rgb: np.ndarray) -> Tuple[float, float]:
    """ คำนวณความสว่างและ Contrast จากช่อง L ในพื้นที่สี LAB """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    return float(np.mean(L)), float(np.std(L))

def image_quality_gate(
    rgb: np.ndarray,
    min_width: int = 224, min_height: int = 224,
    blur_thresh: float = 100.0,      
    brightness_low: float = 50.0,    
    brightness_high: float = 220.0,  
    contrast_low: float = 20.0       
) -> QualityResult:
    """ 
    ตรวจสอบว่าภาพมีความคมชัดและแสงเพียงพอสำหรับการวิเคราะห์หรือไม่ 
    Returns: QualityResult (passed: bool, reasons: list, metrics: dict)
    """
    reasons: List[str] = []
    h, w = rgb.shape[:2]

    # 1. เช็คขนาดภาพ
    if w < min_width or h < min_height: 
        reasons.append(f"resolution_too_small ({w}x{h})")

    # 2. เช็คความเบลอ
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = variance_of_laplacian(gray)
    if blur_score < blur_thresh: 
        reasons.append(f"image_blurry ({blur_score:.1f})")

    # 3. เช็คแสงและ Contrast
    L_mean, L_std = brightness_contrast_lab(rgb)
    if L_mean < brightness_low: reasons.append(f"too_dark ({L_mean:.1f})")
    if L_mean > brightness_high: reasons.append(f"too_bright ({L_mean:.1f})")
    if L_std < contrast_low: reasons.append(f"low_contrast ({L_std:.1f})")

    passed = (len(reasons) == 0)
    return QualityResult(
        passed=passed, 
        reasons=reasons, 
        metrics={"width": w, "height": h, "blur": blur_score, "L_mean": L_mean, "L_std": L_std}
    )