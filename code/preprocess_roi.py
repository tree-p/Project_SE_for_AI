import cv2
import numpy as np

def standardize_roi(
    roi_image: np.ndarray, 
    target_size: tuple = (224, 224), 
    normalization_method: str = 'minmax'
) -> np.ndarray:
    """
    ปรับขนาด (Resize) และค่าพิกเซล (Normalize) ของภาพ ROI
    
    Args:
        roi_image: ภาพเนื้อวัวที่ถูก Crop มาแล้ว (BGR format)
        target_size: ขนาดที่โมเดล AI ต้องการ (กว้าง, ยาว)
        normalization_method: วิธีการปรับค่า ('minmax' หรือ 'zscore')
        
    Returns:
        ภาพที่พร้อมส่งเข้าโมเดล AI (float32)
    """
    
    # 1. Resize: ปรับขนาดภาพให้เป็น 224x224 (ขนาดมาตรฐานของ ImageNet)
    # ใช้ INTER_AREA เพราะมักจะเป็นการย่อขนาดภาพ (Downsampling) ช่วยรักษาความคมชัด
    resized_roi = cv2.resize(roi_image, target_size, interpolation=cv2.INTER_AREA)
    
    # แปลงสีเป็น RGB (โมเดลส่วนใหญ่เทรนด้วย RGB)
    rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
    
    # แปลงชนิดข้อมูลเป็น float32 เพื่อการคำนวณที่แม่นยำ
    processed_roi = rgb_roi.astype(np.float32)

    # 2. Normalization: ปรับสเกลค่าพิกเซล
    if normalization_method == 'minmax':
        # ปรับค่าจาก [0, 255] ให้อยู่ในช่วง [0.0, 1.0]
        processed_roi = processed_roi / 255.0
        
    elif normalization_method == 'zscore':
        # ปรับค่าด้วย ImageNet Mean & Std (ใช้บ่อยกับ Pre-trained models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # ปรับช่วง [0, 1] ก่อน แล้วค่อยทำ Z-score
        processed_roi = processed_roi / 255.0
        processed_roi = (processed_roi - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")
        
    # 3. เตรียมมิติภาพ (เพิ่ม Batch Dimension)
    # โมเดลต้องการ Input เป็น (Batch, Height, Width, Channels) เช่น (1, 224, 224, 3)
    final_tensor = np.expand_dims(processed_roi, axis=0)

    return final_tensor

# --- ตัวอย่างการเรียกใช้งาน (Testing) ---
if __name__ == "__main__":
    # จำลองภาพที่ Crop มาแล้ว (เช่น จากโค้ดของเพื่อน) ขนาด 400x500
    mock_roi = np.random.randint(0, 256, (500, 400, 3), dtype=np.uint8)
    
    # ลองรันฟังก์ชัน
    tensor_input = standardize_roi(mock_roi, target_size=(224, 224), normalization_method='minmax')
    
    print(f"Original ROI shape: {mock_roi.shape}")
    print(f"Model Input shape: {tensor_input.shape}")
    print(f"Min value: {tensor_input.min():.2f}, Max value: {tensor_input.max():.2f}")
    print("✅ พร้อมส่งให้ทำ Classification ต่อแล้ว!")