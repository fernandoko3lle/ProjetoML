#!/usr/bin/env python3
import json, time, numpy as np, torch
import sounddevice as sd
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from transformers import pipeline
from faster_whisper import WhisperModel


# Helper Functions
from Projeto.helper import fusion_emotion

# ---------------- Flask ----------------
app = Flask(__name__, static_url_path="", static_folder=".")
socketio = SocketIO(app, cors_allowed_origins="*")  # websocket em /socket.io/

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# --------------- ML params ------------
FS = 16_000
CHUNK = int(0.25 * FS)
SILENCE_THR = 0.01
SILENCE_MAX = 2          # 0.5 s * 2  => 1 s

ser = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    sampling_rate=FS
)
VAL_SER = {"ang": -1, "sad": -0.5, "neu": 0, "hap": +1}

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = WhisperModel(
    "small", device=device,
    compute_type="float16" if device=="cuda" else "int8"
)
sent = pipeline(
    "text-classification",
    model="cardiffnlp/xlm-roberta-base-tweet-sentiment-pt"
)
VAL_TXT = {"negative": -1, "neutral": 0, "positive": +1}

# ------------- captura de áudio -------------
def capturar():
    print("🎙️  Fale. Pausa de ~1 s encerra o clip. Ctrl-C encerra.")
    clip, silence_cnt = [], 0
    with sd.InputStream(channels=1, samplerate=FS, blocksize=CHUNK) as stream:
        while True:
            data, _ = stream.read(CHUNK)
            rms = np.sqrt(np.mean(data**2))
            clip.append(data)

            silence_cnt = silence_cnt + 1 if rms < SILENCE_THR else 0
            dur = len(clip) * CHUNK / FS

            if silence_cnt >= SILENCE_MAX or dur >= 10:
                all_samples = np.concatenate(clip).flatten()
                clip, silence_cnt = [], 0
                if len(all_samples) < FS:
                    continue

                # ― Emoção de voz ―
                res_a = ser(all_samples, sampling_rate=FS)[0]
                val_a = VAL_SER[res_a["label"]]

                # ― ASR + sentimento ―
                segs, _ = asr.transcribe(all_samples, language="pt", beam_size=1)
                text = "".join(s.text for s in segs).strip()
                if text:
                    res_t = sent(text)[0]
                    val_t = VAL_TXT[res_t["label"]]
                else:
                    res_t = {"label": "neutral", "score": 1.0}
                    val_t = 0

                fusion = fusion_emotion(res_a["label"], val_a, res_t["label"], val_t)

                out = {
                    "audio": {
                        "label": res_a["label"],
                        "confidence": round(res_a["score"], 3),
                        "valence": val_a
                    },
                    "texto": {
                        "transcricao": text,
                        "label": res_t["label"],
                        "confidence": round(res_t["score"], 3),
                        "valence": val_t
                    },
                    "emocao4": fusion
                }
                # envia p/ browser
                
                socketio.emit("sentimento", out, namespace="/")  # broadcast implícito
                socketio.sleep(0)      # rende o controle p/ eventlet/gevent (boa prática)
                # também imprime no terminal (debug)
                print(json.dumps(out, ensure_ascii=False, indent=2))

# -------------- main -------------------
if __name__ == "__main__":
    # inicia captura em thread separada
    socketio.start_background_task(capturar)
    socketio.run(app, host="0.0.0.0", port=5000)
