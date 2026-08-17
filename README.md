# KATHE / KatheBathe

## English → Kashmiri Machine Translation

KATHE / KatheBathe is an English → Kashmiri machine translation model developed by **Muqarab Farooq Vaid and Suhaib Fida** for **KATHE 2026**.

The model is fine-tuned from **`sarvamai/sarvam-translate`** using **QLoRA / LoRA with PEFT**.

<img width="1199" height="555" alt="image" src="https://github.com/user-attachments/assets/e908b0b6-78f5-414e-aa30-3b39c7205e05" />
LINK : https://excalidraw.com/#json=fmEi9loC9z-9HUeORrQ4Y,cBAFupy9-eVpeILLJf1Yjg


---

# Table of Contents

- [Links](#links)
  - [Hugging Face](#hugging-face)
  - [GitHub](#github)
- [How to Run](#how-to-run)
  - [Option 1 - Run from Hugging Face (Preferred)](#option-1---run-from-hugging-face-preferred)
  - [Option 2 - Run from GitHub](#option-2---run-from-github)
- [Automatic Model Download](#automatic-model-download)
- [Model Information](#model-information)
- [Methodology](#methodology)
- [Training Data](#training-data)
- [Installation](#installation)
- [Inference](#inference)
- [Quick Inference Test](#quick-inference-test)
- [Single-Sentence Inference](#single-sentence-inference)
- [Batch Inference](#batch-inference)
- [Kaggle Usage: Also preferred for easy check](#kaggle-usage)
- [Input CSV](#input-csv)
- [Manual CSV](#manual-csv)
- [Custom Text](#custom-text)
- [Inference Flow](#inference-flow)
- [Generation Settings](#generation-settings)
  - [EOS Token - Required Setting](#eos-token---required-setting)
  - [Output Cleanup - Removing Trailing Lines](#output-cleanup---removing-trailing-lines)
- [Technical Details](#technical-details)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Intended Use](#intended-use)
- [Limitations](#limitations)
- [Evaluation](#evaluation)
- [Team](#team)
- [Citations](#citations)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Model Card Authors](#model-card-authors)

---

# Links

## Hugging Face

**Model:**  
https://huggingface.co/KatheBathe/Kathe-Bathe

The Hugging Face repository contains:

- KatheBathe adapter weights
- Tokenizer files
- Model configuration
- `inference.py`
- Model documentation

## GitHub

**Repository:**  
https://github.com/suhaibfida/Kathe-Bathe

The GitHub repository contains:

- `inference.py`
- `requirements.txt`
- `README.md`

The GitHub inference script loads the submitted KatheBathe adapter from Hugging Face.

---

# How to Run

There are two ways to run KatheBathe:

- **Option 1 — Hugging Face (Preferred)**
- **Option 2 — GitHub**

> **Recommended:** We recommend **Option 1 (Hugging Face)** because it downloads the submitted `inference.py` directly from the Hugging Face model repository and uses the submitted model and adapter.

---

# Option 1 - Run from Hugging Face (Preferred)

The preferred method is to download the submitted inference script directly from the KatheBathe Hugging Face repository.

**Hugging Face Model:**

https://huggingface.co/KatheBathe/Kathe-Bathe

## 1. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

If `huggingface_hub` is not already installed:

```bash
pip install -U huggingface_hub
```

## 2. Download `inference.py`

Use `hf_hub_download` to download the exact submitted inference script:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="KatheBathe/Kathe-Bathe",
    filename="inference.py",
    local_dir="/kaggle/working",
    force_download=True,
)

print("inference.py downloaded")
```

This downloads:

```text
KatheBathe/Kathe-Bathe
        ↓
   inference.py
        ↓
/kaggle/working/inference.py
```

## 3. Run the Inference Script

```bash
python /kaggle/working/inference.py
```

The script automatically loads:

- Base model
- KatheBathe adapter
- Tokenizer
- Model configuration

from Hugging Face.

You do **not** need to manually download the model weights.

---

# Option 2 - Run from GitHub

The complete inference code is also available through GitHub.

**GitHub Repository:**

https://github.com/suhaibfida/Kathe-Bathe

## 1. Clone the Repository

```bash
git clone https://github.com/suhaibfida/Kathe-Bathe.git
cd Kathe-Bathe
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Inference Script

```bash
python inference.py
```

The GitHub `inference.py` loads the KatheBathe model and adapter from Hugging Face.

The model weights do not need to be manually copied into the GitHub repository.

---

# Automatic Model Download

You **do not need to manually download the model weights**.

The provided `inference.py` automatically loads the required model components from Hugging Face.

The loading process is:

```text
Run inference.py
       ↓
Load tokenizer
       ↓
Load BF16 base model
       ↓
Load KatheBathe QLoRA adapter
       ↓
Prepare model
       ↓
Generate translation
```

The base model is:

```text
sarvamai/sarvam-translate
```

The submitted adapter is:

```text
KatheBathe/Kathe-Bathe
```

The script automatically downloads the required files if they are not already available locally.

You do not need to manually download:

- Adapter weights
- Adapter configuration
- Tokenizer files
- Model configuration files

The first run may take longer because the model files need to be downloaded.

Subsequent runs can reuse the locally cached files.

> **Internet access is required when the required model files are not already cached locally.**

A merged model is **not required** for the provided inference script.

---

# Model Information

| Configuration | Value |
|---|---|
| **Task** | English → Kashmiri |
| **Base Model** | `sarvamai/sarvam-translate` |
| **Base Architecture** | Gemma 3 4B IT |
| **Fine-tuning** | QLoRA / LoRA |
| **Framework** | PEFT |
| **Inference dtype** | BF16 |
| **Maximum input length** | 1024 |
| **Maximum new tokens** | 232 |
| **Beam size** | 6 |
| **Repetition penalty** | 1.15 |
| **No-repeat n-gram size** | 3 |
| **EOS token ID** | `1` (`<eos>`) |
| **Default batch size** | 16 |
| **License** | GPL-3.0 |

---

# Methodology

KatheBathe was developed using parameter-efficient fine-tuning with **QLoRA / LoRA and PEFT**.

```text
Sarvam-Translate
       ↓
Gemma 3 4B IT
       ↓
QLoRA / LoRA Fine-tuning
       ↓
KatheBathe Adapter
       ↓
English → Kashmiri Translation
```

The base model used for fine-tuning is:

```text
sarvamai/sarvam-translate
```

The trained adapter is loaded on top of the original base model during inference.

A merged model is **not required** for the provided inference script.

---

# Training Data

The model was fine-tuned using the following datasets.

## SMU Qamar - Kashmiri-English Parallel Corpus

Dataset:

https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Parallel-Corpus

## AI4Bharat - BPCC

Dataset:

https://huggingface.co/datasets/ai4bharat/BPCC

---

# Installation

## Requirements

Before running the inference script, you need:

- Python
- pip
- An NVIDIA GPU with BF16 support for the submitted inference configuration
- Internet access for downloading model files if they are not already cached

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
transformers==4.51.3
peft==0.15.2
accelerate
sentencepiece
safetensors
pandas
huggingface_hub
```

The provided inference setup:

- Uses BF16 inference
- Does not use 4-bit `bitsandbytes` quantization
- Uses PEFT for loading the QLoRA adapter
- Uses Hugging Face Hub for downloading the submitted inference script and model resources

---

# Inference

The repository contains a single inference script:

```text
inference.py
```

The script supports:

- Model loading
- Single-sentence inference
- Multiple custom sentences
- Batch CSV inference
- Automatic Kaggle CSV detection
- Manual CSV input
- Output validation
- EOS-controlled generation
- Output cleanup

---

# Quick Inference Test

The quickest way to verify that the **model, tokenizer, adapter, and inference code** are working is:

```bash
python inference.py --text "She was a true visionary."
```

The script will:

1. Load the tokenizer.
2. Load the base model.
3. Load the KatheBathe adapter.
4. Generate a Kashmiri translation.
5. Display the result.

Example:

```text
Input:
She was a true visionary.

Output:
سۄ ٲس اکھ حقیقی بصیرت تھون واجیٚنۍ۔
```

If a translation is generated successfully, the inference setup is working.

---

# Single-Sentence Inference

Translate a single English sentence:

```bash
python inference.py \
    --text "She was a true visionary."
```

You can also provide multiple sentences:

```bash
python inference.py \
    --text "She was a true visionary." \
    --text "The weather is beautiful today."
```

---

# Batch Inference

For batch inference, the input CSV must contain:

```text
ID,sentence
```

Example:

```csv
ID,sentence
1,She was a true visionary.
2,The weather is beautiful today.
3,I like learning new things.
```

Run:

```bash
python inference.py \
    --input /path/to/test.csv \
    --output predictions.csv
```

The output contains:

```text
ID,kashmiri_text
```

Example:

```csv
ID,kashmiri_text
1,سۄ ٲس اکھ حقیقی بصیرت تھون واجیٚنۍ۔
2,...
3,...
```

The script validates:

- Prediction count
- Empty predictions
- ID order
- Output columns
- Saved CSV row count
- Saved CSV IDs
- Empty translations

---

# Kaggle Usage

## 1. Enable GPU

In Kaggle:

```text
Notebook
→ Settings
→ Accelerator
→ GPU
```

Use an NVIDIA GPU with BF16 support.

## 2. Download `inference.py`

The preferred method is to download the exact submitted inference script directly from Hugging Face:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="KatheBathe/Kathe-Bathe",
    filename="inference.py",
    local_dir="/kaggle/working",
    force_download=True,
)

print("inference.py downloaded")
```

This downloads:

```text
KatheBathe/Kathe-Bathe
        ↓
   inference.py
        ↓
/kaggle/working/inference.py
```

## 3. Run

```bash
python /kaggle/working/inference.py
```

The script automatically searches:

```text
/kaggle/input/**/*.csv
```

for compatible CSV files.

The expected input columns are:

```text
ID
sentence
```

If multiple compatible CSV files are found, specify the input manually:

```bash
python /kaggle/working/inference.py \
    --input /kaggle/input/my-dataset/test.csv \
    --output /kaggle/working/predictions.csv
```

---

# Input CSV

The required columns are:

- `ID`
- `sentence`

Example:

```csv
ID,sentence
1,She was a true visionary.
2,The weather is beautiful today.
3,I like learning new things.
```

---

# Manual CSV

Specify the input and output paths:

```bash
python inference.py \
    --input /path/to/test.csv \
    --output /path/to/predictions.csv
```

---

# Custom Text

## Single Sentence

```bash
python inference.py \
    --text "She was a true visionary."
```

## Multiple Sentences

```bash
python inference.py \
    --text "She was a true visionary." \
    --text "The weather is beautiful today."
```

---

# Inference Flow

```text
Load tokenizer
      ↓
Load BF16 base model
      ↓
Load QLoRA adapter
      ↓
One-sentence diagnostic
      ↓
Translate input
      ↓
Validate predictions
      ↓
Save CSV
      ↓
Print first 10 results
      ↓
ALL CHECKS PASSED
```

The one-sentence inference is used as a diagnostic.

The first 10 results printed at the end are previews of predictions that have already been generated.

---

# Generation Settings

```python
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 232

NUM_BEAMS = 6
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3

# Explicit EOS token - required
EOS_TOKEN_ID = 1
```

Generation uses deterministic decoding:

```python
do_sample=False
```

Default batch size:

```text
16
```

If GPU memory is insufficient:

```bash
python inference.py --batch-size 8
```

---

## EOS Token - Required Setting

**`eos_token_id` must be set to `1`, the tokenizer's `<eos>` token.**

Do **not** change it to `106`, which corresponds to `<end_of_turn>`.

The inference configuration uses:

```python
eos_token_id = 1
```

The two settings behave differently:

| Setting | Behavior |
|---|---|
| `eos_token_id = 1` (`<eos>`) | Correct setting used for the submitted model |
| `eos_token_id = 106` (`<end_of_turn>`) | Do not use for the submitted configuration |

In testing, using `<end_of_turn>` caused generation to stop earlier, but resulted in a lower reported score.

Using:

```python
eos_token_id = 1
```

allows beam search to continue evaluating candidate sequences until the actual EOS token is reached.

The trade-off is:

- Longer generation
- Higher compute usage
- More tokens may be generated
- Some raw outputs may contain trailing content

However, this setting produced the better reported score for the submitted configuration.

Therefore:

> **Do not change `eos_token_id = 1` to `106` if you want to reproduce the submitted configuration.**

If runtime is a concern, adjust:

- `--batch-size`
- GPU hardware

rather than changing the EOS token.

---

## Output Cleanup - Removing Trailing Lines

Because generation uses the true `<eos>` token (`1`) rather than `<end_of_turn>` (`106`), the raw decoded output can sometimes contain extra trailing content.

This may include:

- Blank lines
- Stray special tokens
- `<unused...>` tokens
- Repeated content
- Additional generated lines

The inference script handles this through post-processing.

After decoding, the script keeps the first valid non-empty translation line and removes trailing content.

This ensures that:

- Beam search can use the full generation.
- The EOS configuration remains unchanged.
- The final translation is clean.
- The output CSV contains a single translation line.

If you re-implement or modify `inference.py`, keep this cleanup step.

The cleanup step should **not** be replaced by changing the EOS token to `106`.

---

# Technical Details

## Architecture

```text
Gemma 3 4B IT
      ↓
Sarvam-Translate
      ↓
QLoRA Fine-tuning
      ↓
KatheBathe Adapter
```

## Inference Hardware

BF16 inference is designed for an NVIDIA CUDA GPU with BF16 support.

## Software

- Transformers 4.51.3
- PEFT 0.15.2
- Accelerate
- SentencePiece
- Safetensors
- PyTorch
- Hugging Face Hub

---

# Repository Structure

```text
KATHE-KatheBathe/
│
├── inference.py
├── requirements.txt
└── README.md
```

## `inference.py`

The main inference script responsible for:

- Loading the tokenizer
- Loading the base model
- Loading the KatheBathe adapter
- Single-sentence inference
- Batch inference
- Prediction validation
- Saving predictions
- Using the required EOS configuration
- Cleaning trailing generated output

## `requirements.txt`

Contains the Python packages required to run the inference script.

## `README.md`

Contains:

- Model information
- Methodology
- Installation instructions
- Inference instructions
- Dataset information
- Reproducibility information
- EOS configuration
- Generation settings

---

# Reproducibility

The GitHub inference code and Hugging Face model are designed to work together.

```text
GitHub
   │
   └── inference.py
          │
          ▼
Hugging Face
   │
   ├── KatheBathe Adapter
   └── Tokenizer / Configuration
          │
          ▼
sarvamai/sarvam-translate
          │
          ▼
English → Kashmiri
```

The adapter is loaded together with the original base model.

A merged model is not required for the included inference script.

The submitted code can therefore be tested directly against the submitted Hugging Face model weights.

### Important Generation Configuration

To reproduce the submitted inference configuration:

```python
eos_token_id = 1
```

Do not replace it with:

```python
eos_token_id = 106
```

The generation settings should otherwise remain unchanged:

```python
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 232
NUM_BEAMS = 6
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3
do_sample = False
```

---

# Intended Use

KatheBathe is intended for:

- English → Kashmiri translation
- Translation applications
- Websites
- APIs
- Batch translation pipelines
- Research projects
- Evaluation systems
- Kaggle inference workflows

The model should not be treated as:

- A general-purpose factual knowledge model
- An authoritative source of information
- A replacement for human review in high-stakes translation
- A guaranteed dialect or domain specialist

---

# Limitations

Machine translation can produce:

- Incorrect word choices
- Grammar errors
- Contextual mistakes
- Literal translations
- Errors with names and uncommon terminology
- Dialect or spelling variation

For important translations, human review is recommended.

---

# Evaluation

KatheBathe is intended for **English → Kashmiri translation**.

No numerical evaluation results are claimed in this repository because an official evaluation table and test-set results are not provided.

The final competition evaluation may use a private test set.

---

# Team

**KATHE 2026**

- **Muqarab Farooq Vaid**
- **Suhaib Fida**

**Year:** 2026

---

# Citations

If you use KatheBathe or its training resources in research, projects, or tools, please acknowledge the model creators, base model, and datasets.

## KatheBathe

```bibtex
@misc{kathebathe2026,
    title={KATHE / KatheBathe: English-to-Kashmiri Translation Model},
    author={Muqarab Farooq Vaid and Suhaib Fida},
    year={2026},
    publisher={Hugging Face}
}
```

## Sarvam-Translate

Base model:

https://huggingface.co/sarvamai/sarvam-translate

## Gemma 3

```bibtex
@article{gemma_2025,
    title={Gemma 3},
    url={https://arxiv.org/abs/2503.19786},
    publisher={Google DeepMind},
    author={Gemma Team},
    year={2025}
}
```

Paper:

https://arxiv.org/abs/2503.19786

## Kashmiri-English Parallel Corpus

Dataset:

https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Parallel-Corpus

Please cite:

```text
Qumar, S.M.U., Azim, M. & Quadri, S.M.K.
Addressing the data gap: building a parallel corpus for Kashmiri language.
Int. J. Inf. Tecnol. (2024).
https://doi.org/10.1007/s41870-024-01979-8
```

## BPCC / IndicTrans2

Dataset:

https://huggingface.co/datasets/ai4bharat/BPCC

Please cite:

```bibtex
@article{gala2023indictrans,
    title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
    author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh K Khapra and Raj Dabre and Anoop Kunchukuttan},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=vfT4YuzAYA}
}
```

Paper:

https://openreview.net/forum?id=vfT4YuzAYA

---

# License

This model is released under:

```text
GPL-3.0
```

---

# Acknowledgements

We acknowledge the creators of:

- `sarvamai/sarvam-translate`
- `SMUQamar/Kashmiri-English-Parallel-Corpus`
- `ai4bharat/BPCC`
- Gemma 3

These resources were used in developing KatheBathe.

---

# Model Card Authors

**KATHE / KatheBathe**

**Authors:**

- Muqarab Farooq Vaid
- Suhaib Fida

For questions or issues, use the model repository discussion/issues mechanism on Hugging Face.
