#!/usr/bin/env python3
from transformers import pipeline

VALENCE = {"negative": -1, "neutral": 0, "positive": +1}

print("Carregando modelo…")
sent_pipe = pipeline(
    "text-classification",
    model="cardiffnlp/xlm-roberta-base-tweet-sentiment-pt",
)

print("Pronto! Digite frases (ENTER vazio para sair).")
while True:
    txt = input("> ").strip()
    if not txt:
        break
    out = sent_pipe(txt)[0] 
    val = VALENCE[out["label"]]
    print(f"{out['label']:<8} conf={out['score']:.3f}  valence={val:+}")
