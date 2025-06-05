# validation.py

import os
import glob
import pandas as pd
import numpy as np
import soundfile as sf
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from faster_whisper import WhisperModel

from helper import fusion_emotion

# ---------------------- 1) PARÂMETROS GERAIS ----------------------
AUDIO_DIR = "Projeto/db/data16k"
USE_FIRST_N = False
N = 10
OUTPUT_CSV = "fusion_eval_vivae_full.csv"
FS = 16000

# ---------------- 2) INICIALIZAÇÃO DOS MODELOS --------------------
audio_pipeline = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    sampling_rate=FS,
    device=0 if torch.cuda.is_available() else -1
)
VAL_SER = {"ang": -1.0, "sad": -0.5, "neu": 0.0, "hap": +1.0}

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = WhisperModel(
    "small",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)
text_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/xlm-roberta-base-tweet-sentiment-pt",
    device=0 if torch.cuda.is_available() else -1
)
VAL_TXT = {"negative": -1.0, "neutral": 0.0, "positive": +1.0}

# ------------ 3) TOKEN → true_label & audio_lab_truth --------------
TRUE_LABEL_MAP = {
    "achievement": ("alegria", "hap"),
    "triumph":      ("alegria", "hap"),
    "pleasure":     ("alegria", "hap"),
    "anger":        ("raiva",   "ang"),
    "fear":         ("medo",    "sad"),
    "surprise":     ("surpresa","hap")
}
FIXED_TEXT_LABEL = "neutral"

# -------------- 4) VARREDURA E COLETA DE PREVISÕES ----------------
records = []
wav_paths = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
if not wav_paths:
    raise RuntimeError(f"Nenhum arquivo '.wav' encontrado em '{AUDIO_DIR}'")

if USE_FIRST_N:
    wav_paths = wav_paths[:N]
    print(f"==> Rodando apenas nos primeiros {N} arquivos para debug <==")

for i, path in enumerate(wav_paths, start=1):
    fname = os.path.basename(path)
    print(f"[{i}/{len(wav_paths)}] Processando: {fname}")

    fname_lower = fname.lower()
    token_found = next((tok for tok in TRUE_LABEL_MAP if tok in fname_lower), None)
    if token_found is None:
        continue

    true_label, audio_lab_truth = TRUE_LABEL_MAP[token_found]
    text_lab = FIXED_TEXT_LABEL

    wav, sr = sf.read(path)
    if sr != FS:
        wav = np.interp(
            np.linspace(0, len(wav), int(len(wav) * FS / sr)),
            np.arange(len(wav)),
            wav
        ).astype(np.float32)

    try:
        res_a = audio_pipeline(wav, sampling_rate=FS)[0]
        pred_audio_lab = res_a["label"]
    except Exception as e:
        print(f"  Erro no audio_pipeline em '{fname}': {e}")
        continue
    val_a = VAL_SER.get(pred_audio_lab, 0.0)

    segs, _ = asr.transcribe(wav, language="pt", beam_size=1)
    text = "".join(segment.text for segment in segs).strip()
    if text:
        res_t = text_pipeline(text)[0]
        pred_text_lab = res_t["label"]
        val_t = VAL_TXT.get(pred_text_lab, 0.0)
    else:
        pred_text_lab = "neutral"
        val_t = 0.0

    pred_label = fusion_emotion(pred_audio_lab, val_a, pred_text_lab, val_t)

    records.append({
        "file":            fname,
        "true_label":      true_label,
        "audio_lab_truth": audio_lab_truth,
        "audio_lab_pred":  pred_audio_lab,
        "audio_valence":   VAL_SER.get(audio_lab_truth, 0.0),
        "text_lab":        pred_text_lab,
        "text_valence":    val_t,
        "pred_label":      pred_label
    })

if not records:
    raise RuntimeError(f"Nenhum áudio processado em '{AUDIO_DIR}'")

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResultados salvos em '{OUTPUT_CSV}' ({len(df)} linhas)\n")

# -------------------- 5) INSPEÇÃO RÁPIDA --------------------------
print("=== Preview dos resultados ===")
print(df.head().to_string(index=False), "\n")
print("Classes em 'true_label':", sorted(df["true_label"].unique()))
print("Classes em 'pred_label':", sorted(df["pred_label"].unique()), "\n")

# --------------- 6) SPLIT E AVALIAÇÃO (60/20/20) -------------------

def safe_train_test_split(dataframe, stratify_col, test_size, random_state):
    """
    Tenta split estratificado; se falhar por ter classe com <2 exemplos,
    faz split sem estratificação.
    """
    try:
        return train_test_split(
            dataframe, test_size=test_size, random_state=random_state,
            stratify=dataframe[stratify_col]
        )
    except ValueError:
        return train_test_split(
            dataframe, test_size=test_size, random_state=random_state
        )

# split inicial (60% train, 40% temp)
train_df, temp_df = safe_train_test_split(df, "true_label", 0.40, 42)
# split temp (20% val, 20% test)
val_df, test_df = safe_train_test_split(temp_df, "true_label", 0.50, 42)

print("Número de amostras por split:")
print(f"  Train:      {len(train_df)}")
print(f"  Validation: {len(val_df)}")
print(f"  Test:       {len(test_df)}\n")

def evaluate_split(name, split_df):
    print(f"--- {name} Set ---")
    report = classification_report(
        split_df["true_label"],
        split_df["pred_label"],
        labels=["alegria", "raiva", "medo", "surpresa", "indefinido"],
        zero_division=0
    )
    print("Classification Report:\n")
    print(report)
    cm = confusion_matrix(
        split_df["true_label"],
        split_df["pred_label"],
        labels=["alegria", "raiva", "medo", "surpresa", "indefinido"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["alegria", "raiva", "medo", "surpresa", "indefinido"],
        columns=["alegria", "raiva", "medo", "surpresa", "indefinido"]
    )
    print("Confusion Matrix:\n")
    print(cm_df, "\n")

evaluate_split("Train", train_df)
evaluate_split("Validation", val_df)
evaluate_split("Test", test_df)
