
import os
import re
import json
from pathlib import Path
from collections import Counter

import soundfile as sf
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

# 1) Ajuste a pasta onde o VIVAE foi extraído
VIVAE_DIR = Path("../data/VIVAE/Full_set")

# 2) Defina o mapeamento dos Emotion do VIVAE para as labels do pipeline
MAP_VIVAE_TO_PIPELINE = {
    "achievement": "hap",
    "pleasure":    "hap",
    "surprise":    "hap",
    "anger":       "ang",
    "fear":        "sad",
    "pain":        "sad"
}

# 3) Inicialize o pipeline de classificação de áudio (ajuste o checkpoint/modelo conforme seu main2.py)
audio_clf = pipeline(
    task="audio-classification",
    model="superb/hubert-small-superb-er",  # substitua com o modelo que você de fato usa
    device=0 if torch.cuda.is_available() else -1
)

# 4) Função para extrair Emotion (ground truth) e intensidade do nome do arquivo
def parse_filename(filename: str):
    """
    filename ex.: "S04_surprise_peak_10.wav"
    Retorna (emotion, intensity).
    """
    base = Path(filename).stem
    parts = base.split("_")
    if len(parts) >= 3:
        emotion = parts[1]
        intensity = parts[2]
        return emotion, intensity
    return None, None

# 5) Loop para processar cada arquivo e coletar predições versus ground-truth
y_true = []
y_pred = []
detailed_results = []

for wav_path in VIVAE_DIR.glob("*.wav"):
    emotion, intensity = parse_filename(wav_path.name)
    if emotion is None or emotion not in MAP_VIVAE_TO_PIPELINE:
        continue

    gt_label = MAP_VIVAE_TO_PIPELINE[emotion]

    try:
        data, sr = sf.read(str(wav_path))
        if sr != 16000:
            raise ValueError(f"Expected 16 kHz, got {sr} Hz")
        inputs = {'array': data, 'sampling_rate': sr}
        output = audio_clf(inputs)

    except Exception as e:
        print(f"Erro ao processar {wav_path.name}: {e}")
        continue

    pred_label = output[0]["label"]
    pred_score = output[0]["score"]

    y_true.append(gt_label)
    y_pred.append(pred_label)

    detailed_results.append({
        "file": wav_path.name,
        "emotion": emotion,
        "intensity": intensity,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "pred_score": pred_score
    })

# 6) Cálculo de métricas
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)
cm = confusion_matrix(y_true, y_pred, labels=["hap", "neu", "sad", "ang"])

# 7) Exibindo resultados
print(f"Acurácia geral: {acc:.4f}\n")
print("Classification Report:\n", report)
print("Matriz de Confusão (linhas: ground truth; colunas: pred)**:\n", cm)

# 8) (Opcional) Salvar detalhes em um JSON/CSV para posterior análise
with open("val_vivae_detalhado.json", "w", encoding="utf-8") as fout:
    json.dump(detailed_results, fout, ensure_ascii=False, indent=2)
