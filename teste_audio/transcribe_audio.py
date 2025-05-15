#!/usr/bin/env python3
"""
Uso:
    python transcribe_audio.py audios/*.wav
Salva .txt em transcritos/ com mesmo nome-base.
"""
import sys, pathlib, os
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",      # PT-BR ok; troque p/ 'tiny' se CPU fraca
    device="cuda" if torch.cuda.is_available() else "cpu",
    generate_kwargs={"language": "pt"}
)

out_dir = pathlib.Path("../transcritos").resolve()
out_dir.mkdir(exist_ok=True)

for wav in map(pathlib.Path, sys.argv[1:]):
    if not wav.exists():
        print("Arquivo não encontrado:", wav)
        continue

    print(f"🎙️  Transcrevendo {wav.name}…")
    text = asr(wav.as_posix())["text"].strip()
    out_path = out_dir / (wav.stem + ".txt")
    out_path.write_text(text, encoding="utf-8")
    print("→ salvo em", out_path)
