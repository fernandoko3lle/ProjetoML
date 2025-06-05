## Modelos Usados:

* Texto (sentimento): cardiffnlp/xlm-roberta-base-tweet-sentiment-pt

* Áudio (emoções): superb/wav2vec2-base-superb-er


# ProjetoML

Este repositório contém um sistema de inferência em tempo real que combina análise de sentimento de áudio e texto para classificar emoções em quatro categorias (`alegria`, `raiva`, `medo`, `surpresa`) ou `indefinido`. Além disso, inclui um script de validação usando o banco de dados VIVAE para aferir métricas (precision, recall, F1-score) tanto do modelo de áudio quanto do modelo de fusão.

---

## Estrutura de diretórios

```bash
ProjetoML/
├── helper.py               # Funções utilitárias, incluindo fusion_emotion
├── server.py               # Aplicação Flask + SocketIO para inferência em tempo real
├── index.html              # Interface web para exibir resultados em tempo real
├── validation.py           # Script de validação (acurácia, F1, etc.) usando VIVAE
├── requirements.txt        # Dependências do projeto
└── Projeto/
    └── data16k/            # Áudios VIVAE convertidos para 16 kHz (ex.: S01_..._16k.wav)
```

- **helper.py**  
  Contém a função `fusion_emotion(audio_lab, a_val, text_lab, t_val)` que aplica regras para mapear combinações de rótulos e valências de áudio e texto → emoção final (4 classes + `indefinido`).

- **server.py**  
  Aplicação Flask + SocketIO que:
  1. Captura áudio em tempo real pela placa de som.
  2. Executa pipeline de classificação de áudio (Wav2Vec2) → rótulo + valência.
  3. Transcreve com Whisper (`faster-whisper`) → envia para pipeline de sentimento de texto (XLM-RoBERTa-PT) → rótulo + valência.
  4. Chama `fusion_emotion` e envia (via WebSocket) o JSON contendo:
    ```json
     {
       "audio":    {"label": ..., "confidence": ..., "valence": ...},
       "texto":    {"transcricao": ..., "label": ..., "confidence": ..., "valence": ...},
       "emocao4":  "<uma das 4 emoções ou indefinido>"
     }
    ```
  5. A interface `index.html` consome esse WebSocket e exibe os campos em cards.

- **validation.py**  
  Script que percorre todos os arquivos `.wav` em `Projeto/db/data16k/` (VIVAE 16 kHz), extrai o token do nome do arquivo para obter:
  - `true_label` (PT-BR): `alegria`, `raiva`, `medo` ou `surpresa`.
  - `audio_lab_truth` (`hap`, `ang`, `sad`).
  Em seguida, para cada arquivo:
  1. Lê o WAV (já 16 kHz).
  2. Invoca pipeline de áudio (Wav2Vec2) → `pred_audio_lab` + `val_a`.
  3. Executa Whisper → transcrição → pipeline de texto (XLM-RoBERTa-PT) → `pred_text_lab` + `val_t`.
  4. Chama `fusion_emotion(pred_audio_lab, val_a, pred_text_lab, val_t)` → `pred_label`.
  5. Armazena um registro CSV com:
    ```
     file, true_label, audio_lab_truth, audio_lab_pred, audio_valence,
     text_lab, text_valence, pred_label
    ```
  6. Faz split estratificado 60 % train / 20 % val / 20 % test por `true_label` e imprime classification report + confusion matrix (precision, recall, F1) para cada partição.

- **requirements.txt**  
  Lista de pacotes Python necessários (ex.: torch, transformers, faster-whisper, soundfile, scikit-learn, pandas, flask, flask-socketio).

- **Projeto/data16k/**  
  Contém arquivos de áudio `.wav` do banco VIVAE já convertidos para 16 kHz. Exemplo de nome:  
```
  S01_anger_low_01_16k.wav
  S01_fear_strong_05_16k.wav
  S02_pleasure_peak_03_16k.wav
```

---

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://<seu-repositorio>/ProjetoML.git
   cd ProjetoML
   ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Certifique-se de ter os áudios VIVAE em `Projeto/data16k/`. Caso ainda não tenha, copie ou converta os arquivos VIVAE originais para essa pasta.

---

## Como executar o servidor em tempo real

1. **Abra um terminal** na raiz do projeto (onde está `server.py`).

2. Inicie o servidor:
   ```bash
   python server.py
   ```
   - O Flask + SocketIO ficará escutando em `http://0.0.0.0:5000`.

3. **Abra o navegador** em `http://localhost:5000/`.
   - A página `index.html` exibirá quatro cards:  
     - Emoção de áudio (`label`, `confidence`, barra de valência).  
     - Transcrição e sentimento de texto (`label`, `confidence`, barra de valência).  
     - Emoção final (4 classes) calculada pela função `fusion_emotion`.  

4. **Fale no microfone**.
   - O sistema gravará até ~1 s de silêncio consecutivo (ou máximo de 10 s), então enviará o buffer para inferência.
   - Pela saída do terminal, você verá JSONs como:
    ```json
     {
       "audio": {
         "label": "hap",
         "confidence": 0.85,
         "valence": 1.0
       },
       "texto": {
         "transcricao": "eu gosto disso",
         "label": "positive",
         "confidence": 0.92,
         "valence": 1.0
       },
       "emocao4": "alegria"
     }
    ```
   - O navegador atualiza automaticamente (via WebSocket) cada campo em tempo real.

---

## Como validar contra o banco VIVAE

Para rodar **validação completa** e gerar métricas de performance:

1. Confira se **todos** os `.wav` do VIVAE (já convertidos para 16 kHz) estão em `Projeto/data16k/`. No mínimo, os nomes de arquivo devem conter tokens como `anger`, `fear`, `pleasure`, `surprise`, etc.

2. Abra um terminal na raiz do projeto (onde está `validation.py`) e, opcionalmente, ative o virtualenv.

3. Se quiser testar apenas os primeiros 10 arquivos (debug), edite no topo de `validation.py`:
   ```python
   USE_FIRST_N = True
   N = 10
   ```
   Caso contrário, deixe `USE_FIRST_N = False` para processar todos os arquivos.

4. Execute:
   ```bash
   python validation.py
   ```
   - O script exibirá progressão no terminal (`[i/N] Processando: <nome_do_arquivo>.wav`).
   - Ao final, gerará o CSV `fusion_eval_vivae_full.csv` com colunas:
     ```
     file, true_label, audio_lab_truth, audio_lab_pred, audio_valence,
     text_lab, text_valence, pred_label
     ```
   - Em seguida, exibirá:
     - Preview do DataFrame.
     - Quais classes apareceram em `true_label` e `pred_label`.
     - Quantidade de amostras em cada split (Train/Val/Test).
     - **Classification Report** para Train, Validation e Test (precision, recall, F1, support).
     - **Confusion Matrix** (4 classes + `indefinido`).

---

## Estrutura esperada de `fusion_eval_vivae_full.csv`

Cada linha representa um arquivo VIVAE processado:

| file                          | true_label | audio_lab_truth | audio_lab_pred | audio_valence | text_lab | text_valence | pred_label |
|-------------------------------|------------|-----------------|----------------|---------------|----------|--------------|------------|
| S01_anger_low_01_16k.wav      | raiva      | ang             | hap            | -1.0          | neutral  | 0.0          | alegria    |
| S01_fear_peak_02_16k.wav      | medo       | sad             | hap            | -0.5          | neutral  | 0.0          | alegria    |
| S01_pleasure_strong_03_16k.wav| alegria    | hap             | hap            | +1.0          | positive | +1.0         | alegria    |
| S01_surprise_weak_04_16k.wav  | surpresa   | hap             | hap            | +1.0          | neutral  | 0.0          | alegria    |
| …                             | …          | …               | …              | …             | …        | …            | …          |

---

## Como ajustar as regras de fusão

A função `fusion_emotion` está em `helper.py`. Exemplo de versão básica:

```python
def fusion_emotion(audio_lab, a_val, text_lab, t_val):
    if audio_lab == 'sad' and text_lab == 'negative':
        return 'medo'
    if audio_lab == 'hap' and text_lab == 'negative':
        return 'surpresa'
    if audio_lab == 'ang' and text_lab == 'negative':
        return 'raiva'
    if audio_lab == 'hap' and text_lab == 'positive':
        return 'alegria'
    return 'indefinido'
```

Caso você queira que `(audio_lab, neutral)` também seja mapeado, adicione regras como:

```python
if audio_lab == 'hap' and text_lab == 'neutral':
    return 'alegria'
if audio_lab == 'ang' and text_lab == 'neutral':
    return 'raiva'
if audio_lab == 'sad' and text_lab == 'neutral':
    return 'medo'
```

Depois de editar `helper.py`, reexecute `validation.py` para verificar o impacto nas métricas.

---

## Requisitos do sistema

- Python 3.8+
- CPU ou GPU com suporte a PyTorch (para Whisper e XLM-RoBERTa)
- Sistema operacional compatível com `sounddevice` (Linux, macOS ou Windows)
- Internet apenas para baixar modelos (Wav2Vec2, Whisper, XLM-RoBERTa). Depois de baixados, rodar offline.

---
## Resultados do Modelo Testado

Após rodar o script de validação sobre todo o conjunto VIVAE, obtivemos as seguintes métricas no Test Set (180 amostras, 20 % do total):

| Classe     | Precision | Recall | F1-score | Suporte |
|------------|-----------|--------|----------|---------|
| alegria    | 0.40      | 0.84   | 0.54     | 73      |
| raiva      | 0.00      | 0.00   | 0.00     | 35      |
| medo       | 0.17      | 0.03   | 0.05     | 35      |
| surpresa   | 0.00      | 0.00   | 0.00     | 37      |
| indefinido | 0.00      | 0.00   | 0.00     | 0       |
| **acurácia geral** |                         | **0.34** |          | 180     |

**Matriz de Confusão (Test Set)**

|             | alegria | raiva | medo | surpresa | indefinido |
|-------------|:-------:|:-----:|:----:|:--------:|:----------:|
| **alegria**   |   61    |   0   |  5   |    7     |      0     |
| **raiva**     |   26    |   0   |  0   |    9     |      0     |
| **medo**      |   29    |   0   |  1   |    5     |      0     |
| **surpresa**  |   37    |   0   |  0   |    0     |      0     |
| **indefinido**|    0    |   0   |  0   |    0     |      0     |

### Possíveis Causas

1. **Viés do classificador de áudio (Wav2Vec2)**  
   - Muitos arquivos de “raiva” e “medo” foram rotulados como “hap” (happy), de modo que a fusão tende a produzir “alegria” mesmo quando o “true_label” é diferente.

2. **Regras de fusão excessivamente confiantes em `text_lab == 'neutral'`**  
   - A função `fusion_emotion` mapeia `(audio_lab='hap', text_lab='neutral') → alegria` sem levar em conta casos onde “hap” pode corresponder a “surpresa”.  
   - Quando `text_lab` é “neutral”, as classes “raiva”, “medo” e “surpresa” não são priorizadas, gerando muitos falsos positivos em “alegria”.

3. **Ausência de fallback “indefinido”**  
   - A lógica atual nunca retorna “indefinido” para esse conjunto, pois `text_lab` de VIVAE vem quase sempre como “neutral”.  
   - Assim, todo exemplo é classificado em uma das 4 emoções, mesmo que o modelo de fusão não tenha confiança legítima.

### Possíveis Iterações Futuras

1. **Refinar as regras de fusão**  
   - Adicionar tratamento específico para `(audio_lab, text_lab='neutral')`, por exemplo:
     ```python
     if text_lab == 'neutral':
         if audio_lab == 'ang':    return 'raiva'
         if audio_lab == 'sad':    return 'medo'
         if audio_lab == 'hap':    return 'alegria'
     ```
   - Incluir thresholds de valência (`val_a` e `val_t`) para filtrar predicões muito fracas.

2. **Re‐treinar ou substituir o classificador de áudio**  
   - Fine‐tuning de Wav2Vec2 com o conjunto VIVAE para reduzir confusões entre “anger”/“fear” e “hap”.  
   - Avaliar outras arquiteturas (ex.: CNNs sobre espectrogramas) que distingam “raiva” e “medo” de “alegria” com maior precisão.

3. **Introduzir fallback “indefinido” em casos ambíguos**  
   - Se valência do áudio e do texto discordarem fortemente, devolver “indefinido” em vez de forçar uma das 4 emoções.  
   - Exemplo:
     ```python
     if abs(val_a) < threshold and text_lab == 'neutral':
         return 'indefinido'
     ```

4. **Coletar e rotular dados reais de áudio+texto**  
   - Montar um dataset de falas reais (não apenas VIVAE) com transcrições em PT-BR para validar melhor a fusão áudio+texto.  
   - Incluir exemplos de “raiva” e “surpresa” genuínos em contexto de conversação, não apenas expressões isoladas.

5. **Avaliar um modelo multimodal treinado end-to-end**  
   - Investigar arquiteturas que integrem áudio e texto diretamente (p. ex., modelos multimodais baseados em Transformers), em vez de regras manuais.  
   - Explorar frameworks como Hugging Face’s `datasets` e `transformers` para treinar um classificador que receba ambos inputs simultaneamente.

> **Conclusão**:  
> Os resultados atuais indicam uma distribuição fortemente enviesada para “alegria”. Ajustes nas regras de fusão e melhorias no classificador de áudio são as primeiras ações recomendadas. Iterações futuras podem abranger re-treinamento de modelos e coleta de novos dados para suprir as lacunas de performance nas classes “raiva”, “medo” e “surpresa”.  

---

## Como contribuir

1. Crie um *branch* novo para a sua feature:
   ```bash
   git checkout -b minha-feature
   ```
2. Faça alterações e *commits* habitualmente.
3. Abra um *Pull Request* (PR) explicando brevemente as mudanças.
4. Aguarde revisão e aprovação.

---

## Autores
* Fernando Ganzer Koelle
* Roberta Barros
* Gabriel Yamashita

Qualquer dúvida ou problema, abra uma issue ou entre em contato!