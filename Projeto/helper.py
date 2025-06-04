

def fusion_emotion(audio_lab, a_val, text_lab, t_val):
    # 1. Alegria
    if audio_lab == 'hap' or a_val >= 0.5 or text_lab == 'positive' or t_val >= 0.5:
        return 'alegria'
    
    #2. Raiva
    if audio_lab == 'ang' or a_val <= -0.75 or text_lab == 'negative' or t_val <= -0.75:
        return 'raiva'
    
    # 3. Medo
    if audio_lab == 'sad' and a_val == -0.5 and text_lab in ("negative", "neutral"):
        return 'medo'
    
    # 4. Supresa (conflito)
    if abs(a_val - t_val) >= 1:
        return 'surpresa'
    
    # 5. Fallback
    return 'alegria' if a_val > 0 else "raiva" if a_val < 0 else "surpresa"
