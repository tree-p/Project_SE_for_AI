#ปรับแสง
import cv2
import numpy as np

def soft_gray_world_white_balance(rgb: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """ Soft Gray-World White Balance เพื่อปรับสมดุลสีให้สม่ำเสมอ """
    img = rgb.astype(np.float32)
    mean = img.mean(axis=(0, 1))               
    gray_mean = mean.mean()
    full_scale = gray_mean / (mean + 1e-6)     
    scale = 1.0 + alpha * (full_scale - 1.0)
    out = img * scale
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_on_l_channel(rgb: np.ndarray, clip_limit: float = 1.2, tile_grid_size=(8, 8)) -> np.ndarray:
    """ CLAHE on LAB L-channel เพื่อเกลี่ยแสงให้ทั่วถึง """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def preprocess_color_illumination_tuned(
    rgb: np.ndarray,
    use_blur: bool = True,       
    blur_ksize: tuple = (5, 5),  
    use_soft_wb: bool = True,
    wb_alpha: float = 0.25,
    use_clahe: bool = True,
    clahe_clip: float = 1.2
) -> np.ndarray:
    """ 
    Pipeline ปรับปรุงภาพ: 
    1. ลดแสงสะท้อน (Gaussian Blur) 
    2. ปรับ White Balance 
    3. เกลี่ยแสง (CLAHE) 
    """
    out = rgb
    if use_blur:
        out = cv2.GaussianBlur(out, blur_ksize, 0)
    if use_soft_wb:
        out = soft_gray_world_white_balance(out, alpha=wb_alpha)
    if use_clahe:
        out = clahe_on_l_channel(out, clip_limit=clahe_clip)
    return out