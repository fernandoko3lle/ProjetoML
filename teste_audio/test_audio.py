#!/usr/bin/env python3
import sys
from pathlib import Path
from transformers import pipeline
import soundfile as sf

VALENCE = {
    "ang": -1,
    "sad": -0.5,
    "neu": 0,
    "hap": +1,
}

if len(sys.argv) < 2:
    print("Uso: python test_audio.py <arquivo.wav> [outros.wav...]")
    sys.exit(1)

print("⏳ Carregando modelo…")
aud_pipe = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    top_k=None,
)

for wav_path in sys.argv[1:]:
    wav = Path(wav_path)
    if not wav.exists():
        print(f"Arquivo não encontrado: {wav}")
        continue

    # verifica taxa de amostragem
    data, sr = sf.read(wav)
    if sr != 16000:
        print(f"⚠️  {wav.name}: converta para 16 kHz mono para melhor resultado (atual {sr} Hz)")

    res = aud_pipe(wav_path)[0]
    val = VALENCE[res["label"]]
    print(f"{wav.name}: {res['label']:<7} conf={res['score']:.3f}  valence={val:+}")
