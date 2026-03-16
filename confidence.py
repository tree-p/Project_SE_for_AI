#ค่าความมั่นใจ (confidence) ของโมเดล

import numpy as np

def calculate_confidence(probabilities: dict) -> dict:
    # คำนวณ confidence score จาก probabilities
    probs = np.array(list(probabilities.values()))
    
    # Max probability = confidence หลัก
    max_prob = float(np.max(probs))
    
    # Shannon Entropy = ความไม่แน่นอน
    # entropy สูง = model สับสน, entropy ต่ำ = model มั่นใจ
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    max_entropy = float(np.log(len(probs)))  # entropy สูงสุดที่เป็นไปได้
    normalized_entropy = entropy / max_entropy  # 0 = มั่นใจ, 1 = สับสน
    
    # Margin = ช่องว่างระหว่าง top 2 class
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])
    
    # คำนวณ confidence รวม
    confidence_score = round(max_prob * (1 - normalized_entropy * 0.3), 4)
    
    # ระดับความน่าเชื่อถือ
    if confidence_score >= 0.80:
        reliability = "สูง"
        suggestion  = None
    elif confidence_score >= 0.60:
        reliability = "ปานกลาง"
        suggestion  = "ควรตรวจสอบด้วยตนเองเพิ่มเติม"
    else:
        reliability = "ต่ำ"
        suggestion  = "แนะนำให้ถ่ายภาพใหม่หรือตรวจสอบด้วยตนเอง"

    return {
        "confidence_score": confidence_score,
        "reliability":      reliability,
        "suggestion":       suggestion,
        "metrics": {
            "max_probability":    round(max_prob, 4),
            "entropy":            round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "margin":             round(margin, 4)
        }
    }