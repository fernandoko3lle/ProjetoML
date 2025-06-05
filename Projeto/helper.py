TH_SURP = 1.2      # gap mínimo para “surpresa”
TH_AMP_JOY = 0.7   # a_val acima para alegria
TH_AMP_RAGE = -0.7 # a_val abaixo para raiva
TH_TXT_JOY = 0.8   # t_val acima para alegria
TH_TXT_RAGE = -0.8 # t_val abaixo para raiva

def fusion_emotion(a_lab, a_val, t_lab, t_val):
    if abs(a_val - t_val) >= TH_SURP:
        return 'surpresa'
    if a_val <= -0.5 and t_val > -0.2:
        return 'medo'
    if a_val >= TH_AMP_JOY or t_val >= TH_TXT_JOY:
        return 'alegria'
    if a_val <= TH_AMP_RAGE or t_val <= TH_TXT_RAGE:
        return 'raiva'
    # fallback
    if a_val > 0:
        return 'alegria'
    if a_val < 0:
        return 'raiva'
    return 'surpresa'
