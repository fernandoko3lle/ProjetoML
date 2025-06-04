#!/usr/bin/env python3
import sounddevice as sd, numpy as np, torch, json, time
from transformers import pipeline
from faster_whisper import WhisperModel

# Helper Functions

FS = 16_000
CHUNK = int(0.25 * FS)             # 0.5 s
SILENCE_THR = 0.01                # RMS abaixo disto é “silêncio”
SILENCE_MAX = 2      # 0.8 s = 2 chunks

# ----- modelos -----
ser = pipeline("audio-classification",
               model="superb/wav2vec2-base-superb-er",
               sampling_rate=FS)
VAL_SER = {"ang": -1, "sad": -0.5, "neu": 0, "hap": +1}

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = WhisperModel("small", device=device, compute_type="float16" if device=="cuda" else "int8")
sent = pipeline("text-classification",
                model="cardiffnlp/xlm-roberta-base-tweet-sentiment-pt")
VAL_TXT = {"negative": -1, "neutral": 0, "positive": +1}

# ----- captura -----
print("🎙️  Fale. Pausa de ~1 s encerra o clip. Ctrl-C sai.")
clip = []
silence_cnt = 0

with sd.InputStream(channels=1, samplerate=FS, blocksize=CHUNK) as stream:
    try:
        while True:
            data, _ = stream.read(CHUNK)
            rms = np.sqrt(np.mean(data**2))
            clip.append(data)

            if rms < SILENCE_THR:
                silence_cnt += 1
            else:
                silence_cnt = 0

            # clip encerra se silêncio prolongado OU 10 s contínuos
            dur = len(clip) * 0.5
            if silence_cnt >= SILENCE_MAX or dur >= 10:
                all_samples = np.concatenate(clip).flatten()
                clip = []; silence_cnt = 0
                if len(all_samples) < FS:        # ignorar clipes <1 s
                    continue

                # --- Emoção de voz ---
                res_a = ser(all_samples, sampling_rate=FS)[0]
                val_a = VAL_SER[res_a["label"]]

                # --- ASR + sentimento ---
                segs, _ = asr.transcribe(all_samples, language="pt", beam_size=1)
                text = "".join(s.text for s in segs).strip()
                if text:
                    res_t = sent(text)[0]
                    val_t = VAL_TXT[res_t["label"]]
                else:
                    res_t = {"label": "neutral", "score": 1.0}
                    val_t = 0


                out = {
                    "audio": {"label": res_a["label"], "conf": round(res_a["score"],3), "valence": val_a},
                    "texto": {"transcricao": text, "label": res_t["label"], "conf": round(res_t["score"],3), "valence": val_t},
                }
                

                print(json.dumps(out, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        print("\nFim.")
