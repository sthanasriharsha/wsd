<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=140&section=header&text=Word%20Sense%20Disambiguation&fontSize=32&fontColor=ffffff&animation=fadeIn&fontAlignY=55" width="100%"/>

<h3>🌐 Specific Word Scorer & Sense Analyzer</h3>
<p><em>Context-aware WSD pipeline powered by ConSeC + WordNet · Deployed on Hugging Face Spaces</em></p>

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-FFD21F?style=for-the-badge)](https://huggingface.co/spaces/Harsha909/wsd-scorer)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey?style=for-the-badge)](LICENSE.txt)
[![EMNLP 2021](https://img.shields.io/badge/Paper-EMNLP%202021-blue?style=for-the-badge)](https://aclanthology.org/2021.emnlp-main.112/)

</div>

---

## 📖 What This Project Does

This project builds an end-to-end **Word Sense Disambiguation (WSD)** pipeline that processes any English sentence and, for **every single word** in it:

1. **Disambiguates the correct sense** of the word using the **ConSeC model** (EMNLP 2021)
2. **Fetches specific/hyponym words** from **WordNet** based on that exact sense
3. **Scores each specific word** with ConSeC to rank how well it fits in the original context
4. **Returns ranked output** in both JSON and CSV format

> **Example:**
> Input: `"I deposited money in the bank"`
> For `"bank"` → ConSeC picks sense: *financial institution* → hyponyms scored:
> `savings bank (0.91)`, `credit union (0.80)`, `commercial bank (0.74)` ...
> Output key: `bank_savingsBank`

This is **not** a basic WSD tool — it goes further by scoring *how specifically* each word is used in its context, using a feedback-loop architecture from the ConSeC paper.

---

## 🧠 How the Pipeline Works

```
Input Sentence
      │
      ▼
┌─────────────────────────────┐
│ Step 1: Tokenize + POS Tag  │  ← spaCy
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 2: Detect Ambiguous    │  ← words with 2+ WordNet senses
│         Content Words       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 3: ConSeC Sense        │  ← picks correct synset per word
│         Disambiguation      │    e.g. bank → bank.n.01
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 4: WordNet Hyponyms    │  ← fetches specific words for
│         (Specific Words)    │    the disambiguated sense only
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 5: Score Specifics     │  ← ConSeC scores each hyponym
│         with ConSeC         │    in the original sentence context
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 6: Save Output         │  ← JSON + CSV files
│   word_bestSpecific: [...]  │
└─────────────────────────────┘
```

**Function words** (`the`, `in`, `I`, `is`, etc.) are handled by a built-in fallback table with grammatical labels (e.g. `"the"` → `definite article (1.00)`), ensuring complete coverage for every token.

---

## ✨ Features

- ✅ Processes **every token** in a sentence — content words AND function words
- ✅ Uses **ConSeC feedback-loop** for sense selection before fetching hyponyms (prevents wrong-sense hyponyms)
- ✅ Scores hyponyms **in context** — not just any hyponym, but the one that fits *this* sentence
- ✅ **Web UI** via Flask (runs locally or on Hugging Face Spaces at port 7860)
- ✅ Supports **single sentence**, **.txt file** (one sentence per line), and **.csv file** inputs
- ✅ Outputs: **scored JSON**, **clean JSON**, and **CSV** with full per-word data
- ✅ Dockerized for easy deployment

---

## 📁 Project Structure

```
wsd/
├── wsd_pipeline.py       # Core WSD logic: tokenization, POS tagging, ConSeC disambiguation
├── score.py              # Specific word scorer: hyponym fetching + ConSeC scoring + output saving
├── flask_app.py          # Flask web UI: 3 input modes, results display, file download
├── setup.sh              # Conda environment setup script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container setup for Hugging Face deployment
├── conf/                 # Hydra YAML configuration files for training & evaluation
├── data/                 # WSD Evaluation Framework datasets + Wikipedia PMI data
├── experiments/          # Model checkpoints (ConSeC-SemCor, ConSeC-SemCor+WNGT)
└── src/
    └── scripts/
        └── model/
            ├── train.py              # Training script (PyTorch Lightning + Hydra)
            ├── raganato_evaluate.py  # Evaluation against Raganato benchmark
            └── predict.py            # Interactive predict script
```

---

## 🚀 Quick Start

### Prerequisites
- Debian/Ubuntu system
- [conda](https://docs.conda.io/en/latest/) installed

### 1. Setup Environment

```bash
git clone https://github.com/sthanasriharsha/wsd.git
cd wsd
bash setup.sh
```

### 2. Download Required Data

Download [Wikipedia Freqs (PMI data)](https://drive.google.com/file/d/1WqNKZZFXM1xrVlDUOFSwMBINJGFlbM_l/view?usp=sharing) and extract:

```bash
cd data/
tar -xvf pmi.tar.gz
rm pmi.tar.gz
cd ..
```

### 3. Download Model Checkpoints

Place inside `experiments/released-ckpts/`:

| Checkpoint | Training Data | Score (ALL) |
|---|---|---|
| [ConSeC-SemCor](https://drive.google.com/file/d/15__onFMnfGKKyulFxQLStUxdNKiqq-Rn/view?usp=sharing) | SemCor | 82.0 |
| [ConSeC-SemCor+WNGT](https://drive.google.com/file/d/1dwzQ7QDwe8hH4pGBBe-5g4N_BI2eLDfA/view?usp=sharing) | SemCor + WNGT | 83.2 |

### 4. Run the Web UI

```bash
# Set PYTHONPATH and start Flask
PYTHONPATH=$(pwd) python flask_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 🖥️ Web Interface

The Flask app (`flask_app.py`) provides three input modes:

| Mode | Description |
|------|-------------|
| **Single Sentence** | Type any sentence directly into the text box |
| **Upload .txt** | One sentence per line; lines starting with `#` are skipped |
| **Upload .csv** | First column of sentences; header row auto-skipped |

**Output:** Results are shown live on the page with per-word scored pills, plus downloadable **CSV** and **JSON** files.

---

## 📊 Output Format

### JSON (Clean)
```json
{
  "I deposited money in the bank": {
    "I": ["first person singular (1.00)"],
    "deposited_put": ["put (0.93)", "placed (0.89)", "invested (0.85)"],
    "money_cash": ["cash (0.96)", "currency (0.91)", "funds (0.87)"],
    "in": ["locative preposition (1.00)"],
    "the": ["definite article (1.00)"],
    "bank_savingsBank": ["savings bank (0.91)", "credit union (0.80)", "commercial bank (0.74)"]
  }
}
```

> **Key format:** `originalWord_bestSpecificWord` (best specific word in camelCase)

### CSV Columns

| Column | Description |
|--------|-------------|
| `sentence_id` | Sentence number |
| `original_word` | Token (e.g. `bank`) |
| `label` | Output key (e.g. `bank_savingsBank`) |
| `best_specific` | Top-scored specific word |
| `best_score` | Score of top specific (0.0–1.0) |
| `chosen_sense` | WordNet definition ConSeC selected |
| `used_synset` | Synset name (e.g. `bank.n.01`) |
| `all_specifics` | Pipe-separated scored list |

---

## 🔬 Interactive Predict (CLI)

Test the raw ConSeC model interactively from the terminal:

```bash
PYTHONPATH=$(pwd) python src/scripts/model/predict.py \
  experiments/released-ckpts/consec_semcor_normal_best.ckpt -t
```

```
Enter space-separated text: I have a beautiful dog
Target position: 4
Enter candidate lemma-def pairs. " --- " separated. Enter to stop
 * dog --- a member of the genus Canis
 * dog --- someone who is morally reprehensible

        # predictions
                 * 0.9939   dog   a member of the genus Canis
                 * 0.0061   dog   someone who is morally reprehensible
```

---

## 📐 Train & Evaluate

### Training

```bash
PYTHONPATH=$(pwd) python src/scripts/model/train.py
```

Training parameters are managed via [Hydra](https://hydra.cc/) YAML files in `conf/`. See [Hydra tutorials](https://hydra.cc/docs/tutorials/intro/) to customize datasets, optimizer, and model settings.

### Evaluation (Raganato Framework)

```bash
PYTHONPATH=$(pwd) python src/scripts/model/raganato_evaluate.py \
  model.model_checkpoint=experiments/released-ckpts/consec_semcor_normal_best.ckpt \
  test_raganato_path=data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007
```

> `test_raganato_path` expects a prefix such that both `{path}.data.xml` and `{path}.gold.key.txt` exist.

---

## 🐳 Docker

```bash
docker build -t wsd-pipeline .
docker run -p 7860:7860 wsd-pipeline
```

---

## 📚 Based On

This project builds on the **ConSeC** model, accepted at **EMNLP 2021**:

> Barba et al., *"ConSeC: Word Sense Disambiguation as Continuous Sense Comprehension"*, EMNLP 2021.
> [Paper](https://aclanthology.org/2021.emnlp-main.112)

```bibtex
@inproceedings{barba-etal-2021-consec,
    title     = "{C}on{S}e{C}: Word Sense Disambiguation as Continuous Sense Comprehension",
    author    = "Barba, Edoardo and Procopio, Luigi and Navigli, Roberto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year      = "2021",
    url       = "https://aclanthology.org/2021.emnlp-main.112",
    pages     = "1492--1503",
}
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![WordNet](https://img.shields.io/badge/NLTK%20WordNet-154F5B?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21F?style=flat-square&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-Config-89B4FA?style=flat-square)

---

## 📜 License

This work is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
For commercial use, please contact the author.

---

## 👤 Author

**G Sthana Sriharsha** — AI/ML Intern @ C-DAC Thiruvananthapuram

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-1a1a2e?style=flat-square)](https://sthanasriharsha.github.io/portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/gundumalla-sthana-sriharsha-a23472300/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/sthanasriharsha)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=100&section=footer" width="100%"/>

</div>
