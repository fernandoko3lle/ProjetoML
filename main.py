#!/usr/bin/env python3
"""
Uso:
    python main.py caminho/arquivo.wav
Requisitos já instalados na venv:
    transformers torch soundfile openai-whisper
"""
from pathlib import Path
import sys, json, torch
from transformers import pipeline

# ---------- modelos ----------
print("⏳ Carregando modelos…")
asr_pipe   = pipeline("automatic-speech-recognition",
                      model="openai/whisper-small",
                      device="cuda" if torch.cuda.is_available() else "cpu",
                      generate_kwargs={"language": "pt"})
audio_pipe = pipeline("audio-classification",
                      model="superb/wav2vec2-base-superb-er")
text_pipe  = pipeline("text-classification",
                      model="cardiffnlp/xlm-roberta-base-tweet-sentiment-pt")

VALENCE_AUDIO = {"ang": -1, "sad": -0.5, "neu": 0, "hap": +1}
VALENCE_TEXT  = {"negative": -1, "neutral": 0, "positive": +1}

# ---------- entrada ----------
if len(sys.argv) != 2:
    sys.exit("Uso: python main.py arquivo.wav")

wav_path = Path(sys.argv[1])
if not wav_path.exists():
    sys.exit(f"Arquivo não encontrado: {wav_path}")

# ---------- emoção no áudio ----------
aud_res = audio_pipe(wav_path.as_posix())[0]        # top-1
aud_label, aud_conf = aud_res["label"], aud_res["score"]
aud_val = VALENCE_AUDIO[aud_label]

# ---------- ASR ----------
transcrito = asr_pipe(wav_path.as_posix())["text"].strip()

# ---------- sentimento no texto ----------
txt_res = text_pipe(transcrito)[0]
txt_label, txt_conf = txt_res["label"], txt_res["score"]
txt_val = VALENCE_TEXT[txt_label]

# ---------- saída agregada ----------
output = {
    "audio": {
        "label": aud_label,
        "confidence": round(aud_conf, 3),
        "valence": aud_val,
    },
    "texto": {
        "transcricao": transcrito,
        "label": txt_label,
        "confidence": round(txt_conf, 3),
        "valence": txt_val,
    },
}
print(json.dumps(output, ensure_ascii=False, indent=2))

