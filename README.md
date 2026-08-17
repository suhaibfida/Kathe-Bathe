# KATHE / KatheBathe

**English → Kashmiri Machine Translation**

KATHE / KatheBathe is an English → Kashmiri machine translation model developed by **Muqarab Farooq Vaid and Suhaib Fida** for **KATHE 2026**.

The model is fine-tuned from **`sarvamai/sarvam-translate`** using **QLoRA / LoRA with PEFT**.

---

## 🔗 Links & How to Run

### 💻 Run from GitHub

**GitHub Repository:**
`<YOUR_GITHUB_REPOSITORY_URL>`

The GitHub repository contains the inference code and requirements.

To run the model from GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KATHE-KatheBathe
pip install -r requirements.txt
python inference.py --text "She was a true visionary."
```

The `inference.py` script automatically loads the required model and adapter from Hugging Face.

---

### 🤗 Run from Hugging Face

**Hugging Face Model:**
https://huggingface.co/KatheBathe/Kathe-Bathe

The Hugging Face repository contains the submitted **KatheBathe adapter weights**, tokenizer/configuration files, `inference.py`, and model documentation.

You can download `inference.py` directly from the Hugging Face repository and run it locally.

```bash
python inference.py --text "She was a true visionary."
```

The required model files are downloaded automatically from Hugging Face when the script runs.

---

### 🔄 GitHub + Hugging Face

The two repositories work together:

```text
             GitHub
                │
                │ inference.py
                ▼
       Load inference code
                │
                ▼
            Hugging Face
                │
        ┌───────┴────────┐
        │                │
  Base Model        KatheBathe
  Sarvam-Translate    Adapter
        │                │
        └───────┬────────┘
                ▼
        English → Kashmiri
```

**GitHub provides the code.**

**Hugging Face provides the submitted model adapter and model files.**

You can therefore start from either repository:

* **GitHub:** clone the repository and run `inference.py`.
* **Hugging Face:** download `inference.py` and use the model repository directly.

No manual download of the adapter weights is required when using the provided inference script.

---

## Table of Contents

* [Model](#model)
* [Methodology](#methodology)
* [Training Data](#training-data)
* [Installation](#installation)
* [Model Loading and Automatic Download](#model-loading-and-automatic-download)
* [Inference](#inference)
* [Quick Inference Test](#quick-inference-test)
* [Single-Sentence Inference](#single-sentence-inference)
* [Batch Inference](#batch-inference)
* [Kaggle Usage](#kaggle-usage)
* [Generation Configuration](#generation-configuration)
* [Inference Pipeline](#inference-pipeline)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Reproducibility](#reproducibility)
* [Limitations](#limitations)
* [Team](#team)
* [License](#license)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

---

# Model

| Property              | Details                     |
| --------------------- | --------------------------- |
| **Task**              | English → Kashmiri          |
| **Base Model**        | `sarvamai/sarvam-translate` |
| **Base Architecture** | Gemma 3 4B IT               |
| **Fine-tuning**       | QLoRA / LoRA                |
| **Framework**         | PEFT                        |
| **Inference dtype**   | BF16                        |
| **Year**              | 2026                        |

### Model Weights

The trained KatheBathe adapter is hosted on:

**https://huggingface.co/KatheBathe/Kathe-Bathe**

A merged model is **not required** for inference.

---

# Methodology

The model was developed using parameter-efficient fine-tuning with **QLoRA**.

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

QLoRA was used to fine-tune the model while keeping the base model frozen and training a smaller set of additional parameters.

---

# Training Data

The model was fine-tuned using the following datasets.

## Kashmiri-English Parallel Corpus

**`SMUQamar/Kashmiri-English-Parallel-Corpus`**

https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Parallel-Corpus

## BPCC

**`ai4bharat/BPCC`**

https://huggingface.co/datasets/ai4bharat/BPCC

---

# Installation

## Requirements

Before running the inference script, make sure you have:

* **Python 3.12 or newer**
* **Git**
* An **NVIDIA GPU with BF16 support** for the submitted inference configuration

## 1. Clone the GitHub Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KATHE-KatheBathe
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
python inference.py --text "She was a true visionary."
```

---

# Model Loading and Automatic Download

The provided `inference.py` handles model loading automatically.

You **do not need to manually download the model weights**.

When the script is run, it loads:

```text
Tokenizer
    ↓
sarvamai/sarvam-translate
    ↓
KatheBathe Adapter
    ↓
Inference
```

If the required files are not already available locally, they are downloaded automatically from Hugging Face.

The first run may take longer because the model files need to be downloaded.

Subsequent runs can reuse the locally cached files.

> **Internet access is required when the required model files are not already available in the local Hugging Face cache.**

---

# Inference

The repository contains a single inference script:

```text
inference.py
```

It supports:

* Model loading
* Single-sentence inference
* Multiple text inputs
* Batch CSV inference
* Kaggle inference
* Output validation

---

# Quick Inference Test

The quickest way to verify that the **code, tokenizer, base model, adapter, and generation process** are working is:

```bash
python inference.py --text "She was a true visionary."
```

The script will load the model and generate a Kashmiri translation.

Example:

```text
Input:
She was a true visionary.

Translation:
سۄ ٲس اکھ حقیقی بصیرت تھون واجیٚنۍ۔
```

If a translation is generated successfully, the inference setup is working.

---

# Single-Sentence Inference

Translate one sentence:

```bash
python inference.py \
    --text "She was a true visionary."
```

Multiple sentences:

```bash
python inference.py \
    --text "She was a true visionary." \
    --text "The weather is beautiful today."
```

---

# Batch Inference

For batch translation, provide a CSV containing:

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

Output:

```text
ID,kashmiri_text
```

The script validates:

* Prediction count
* Empty predictions
* ID order
* Output columns
* Saved CSV row count
* Saved CSV IDs
* Empty translations

---

# Kaggle Usage

Enable a GPU:

```text
Notebook
→ Settings
→ Accelerator
→ GPU
```

Then:

```bash
python inference.py
```

The script can automatically search:

```text
/kaggle/input/**/*.csv
```

for a compatible CSV.

Expected columns:

```text
ID
sentence
```

For a specific file:

```bash
python inference.py \
    --input /kaggle/input/my-dataset/test.csv \
    --output /kaggle/working/predictions.csv
```

---

# Generation Configuration

The submitted inference configuration uses:

```text
Maximum input length:     1024
Maximum new tokens:       232
Beam size:                6
Repetition penalty:       1.15
No-repeat n-gram size:    3
Sampling:                 Disabled
Default batch size:       16
Inference dtype:          BF16
```

Generation uses:

```python
do_sample=False
```

If GPU memory is insufficient:

```bash
python inference.py --batch-size 8
```

---

# Inference Pipeline

```text
                 Input
                   │
          ┌────────┴────────┐
          │                 │
    Single Text          CSV Batch
          │                 │
          └────────┬────────┘
                   ↓
            Load tokenizer
                   ↓
             Load base model
                   ↓
          Load KatheBathe adapter
                   ↓
          Generate translation
                   ↓
            Validate output
                   ↓
             Save predictions
```

---

# Repository Structure

```text
KATHE-KatheBathe/
│
├── inference.py
├── README.md
└── requirements.txt
```

### `inference.py`

The main inference script responsible for:

* Loading the tokenizer
* Loading the base model
* Loading the KatheBathe adapter
* Single-sentence inference
* Batch inference
* Prediction validation
* Saving the output CSV

### `README.md`

Contains the methodology, model information, installation instructions, inference instructions, and reproducibility information.

### `requirements.txt`

Contains the Python dependencies required to run the inference script.

---

# Requirements

The main Python dependencies are:

```text
torch
transformers==4.51.3
peft==0.15.2
accelerate
sentencepiece
safetensors
pandas
huggingface_hub
```

Install them with:

```bash
pip install -r requirements.txt
```

### Hardware

The submitted configuration uses BF16 inference and is intended for an NVIDIA GPU with BF16 support.

---

# Reproducibility

The submitted GitHub code and Hugging Face model are designed to work together.

```text
GitHub
   │
   │ inference.py
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

The inference script automatically retrieves the required model files from Hugging Face when they are not already cached locally.

This allows the submitted code to be tested directly against the submitted model weights.

---

# Limitations

The model may produce:

* Incorrect word choices
* Grammar errors
* Contextual mistakes
* Literal translations
* Errors with names and uncommon terminology
* Dialect or spelling variation

For important translations, human review is recommended.

---

# Team

**KATHE 2026**

* **Muqarab Farooq Vaid**
* **Suhaib Fida**

**Year:** 2026

---

# License

The model is released under the **GPL-3.0** license.

---

# Citation

If you use KatheBathe, please cite:

```bibtex
@misc{kathebathe2026,
    title={KATHE / KatheBathe: English-to-Kashmiri Translation Model},
    author={Muqarab Farooq Vaid and Suhaib Fida},
    year={2026},
    publisher={Hugging Face}
}
```

---

# Acknowledgements

We acknowledge the creators of:

* `sarvamai/sarvam-translate`
* `SMUQamar/Kashmiri-English-Parallel-Corpus`
* `ai4bharat/BPCC`
* Gemma 3

These resources were used in developing KatheBathe.
