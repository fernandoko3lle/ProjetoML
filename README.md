# ProjetoML_sarcasmo
Projeto para tentativa de medir o sarcasmo a partir de modelos ja treinados de audio e texto 

## Modelos Usados:

* Texto (sentimento): cardiffnlp/xlm-roberta-base-tweet-sentiment-pt

* Áudio (emoções): superb/wav2vec2-base-superb-er



## 🚀 Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/<SEU_USUARIO>/sentimetria-live.git
cd sentimetria-live

# 2. Ambiente virtual
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Dependências
pip install -r backend/requirements.txt
```

## Se possuir GPU NVIDIA (CUDA 11.8+):

* Mude no requirements.txt
```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.2.2+cu118 torchaudio==2.2.2+cu118 \
            torchvision==0.18.0+cu118 \
            -f https://download.pytorch.org/whl/torch_stable.html
pip install "faster-whisper[cuda]==1.0.1"
```