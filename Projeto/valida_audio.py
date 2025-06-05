
import os
import json
import sys
from pathlib import Path

# 1) Ajustar o PYTHONPATH para conseguir importar server e helper
proj_dir = Path(__file__).parent.resolve().parent.resolve()
sys.path.append(str(proj_dir))

from server import ser

# 4) Dependências para métricas
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score
)

# 5) Definir onde está o VIVAE (ajuste o caminho conforme a sua organização de pastas)
print(proj_dir)
VIVAE_DIR = proj_dir / "data" / "VIVAE" / "Full_set"

# 6) Mapeamento emotion -> label do pipeline de áudio (mesmo usado antes)
MAP_VIVAE_TO_PIPELINE = {
    "achievement": "hap",
    "pleasure":    "hap",
    "surprise":    "hap",
    "anger":       "ang",
    "fear":        "sad",
    "pain":        "sad"
}

# Função para extrair “emotion” e “intensity” do nome de arquivo
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

def main():
    y_true = []
    y_pred = []
    detalhes = []

    # -----------------------------------------------------------
    # 1) Verifica se a pasta existe
    if not VIVAE_DIR.exists():
        print(f"ERRO: diretório VIVAE não encontrado em: {VIVAE_DIR}")
        print("→ Ajuste o caminho da variável VIVAE_DIR no script.")
        return

    # -----------------------------------------------------------
    # 2) Percorre cada arquivo .wav no VIVAE
    for wav_path in sorted(VIVAE_DIR.glob("*.wav")):
        emotion, intensity = parse_filename(wav_path.name)
        if emotion is None or emotion not in MAP_VIVAE_TO_PIPELINE:
            continue

        # 2.1) Rótulo “ground truth” no formato do pipeline
        gt_label = MAP_VIVAE_TO_PIPELINE[emotion]

        # 2.2) Executa o pipeline de áudio que está definido em server.py
        #     ATENÇÃO: o pipeline `ser` em server.py já foi inicializado com:
        #       ser = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", sampling_rate=FS)
        #
        #     A chamada correta é: ser( {"array": array_de_amostras, "sampling_rate": FS} )[0]
        #     Mas, no server.py, eles usam diretamente a chamada: ser(all_samples, sampling_rate=FS)
        #
        #     Para forçar o mesmo comportamento, vamos ler o .wav em numpy via soundfile
        try:
            import soundfile as sf
        except ImportError:
            print("Por favor, instale `soundfile` (pip install soundfile) para ler .wav.")
            return

        try:
            data, sr = sf.read(str(wav_path))
        except Exception as e:
            print(f"Erro ao ler `{wav_path.name}`: {e}")
            continue

        # 2.3) Chama exatamente como em server.py: 
        try:
            FS = 44100
            result_audio = ser(data, sampling_rate=FS)
        except Exception as e:
            print(f"Erro ao chamar `ser` para `{wav_path.name}`: {e}")
            continue

        # result_audio é uma lista de dicionários, pegamos o primeiro:
        pred_label = result_audio[0]["label"]
        pred_score = result_audio[0]["score"]

        # 2.4) Acumula para métricas
        y_true.append(gt_label)
        y_pred.append(pred_label)

        detalhes.append({
            "file": wav_path.name,
            "emotion": emotion,
            "intensity": intensity,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "pred_score": pred_score
        })

    # =================================================================================
    # 3) Se não houver amostras, encerra
    if len(y_true) == 0:
        print("Nenhuma amostra válida encontrada. Verifique VIVAE_DIR e sampling rate.")
        return

    # =================================================================================
    # 4) Cálculo das métricas
    labels = ["hap", "neu", "sad", "ang"]  # ordem fixa para confusion matrix

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)

    # Relatório por classe (precision/recall/f1/support)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report_dict).T
    df_report = df_report[["precision", "recall", "f1-score", "support"]]

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # =================================================================================
    # 5) Exibe no terminal
    print("\n===== MÉTRICAS GLOBAIS =====")
    print(f"Acurácia:             {acc:.4f}")
    print(f"Acurácia balanceada:  {bal_acc:.4f}")
    print(f"Cohen's Kappa:        {kappa:.4f}")
    print(f"Precision (macro):    {prec_macro:.4f}")
    print(f"Recall (macro):       {rec_macro:.4f}")
    print(f"F1‐score (macro):     {f1_macro:.4f}")
    print(f"F1‐score (micro):     {f1_micro:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(df_report.to_string(float_format='%.4f'))

    print("\n===== MATRIZ DE CONFUSÃO =====")
    print(df_cm.to_string())

    # =================================================================================
    # 6) Salva relatórios em arquivos na pasta atual (“Projeto/”)
    df_report.to_csv(proj_dir / "evaluation_classification_report.csv", index=True, float_format="%.4f")
    df_cm.to_csv(proj_dir / "evaluation_confusion_matrix.csv", index=True)
    metrics_globais = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
    with open(proj_dir / "evaluation_global_metrics.json", "w", encoding="utf-8") as fout:
        json.dump(metrics_globais, fout, ensure_ascii=False, indent=2)

    #    – Detalhamento de cada arquivo
    with open(proj_dir / "evaluation_detalhado.json", "w", encoding="utf-8") as fout:
        json.dump(detalhes, fout, ensure_ascii=False, indent=2)

    print("\nRelatórios gerados em:")
    print(f" • {proj_dir/'evaluation_classification_report.csv'}")
    print(f" • {proj_dir/'evaluation_confusion_matrix.csv'}")
    print(f" • {proj_dir/'evaluation_global_metrics.json'}")
    print(f" • {proj_dir/'evaluation_detalhado.json'}")

if __name__ == "__main__":
    main()
