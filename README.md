# Kathe-Bathe

**English → Kashmiri Machine Translation**

KATHE / KatheBathe is an English → Kashmiri translation model developed by **Muqarab Farooq Vaid and Suhaib Fida** for **KATHE 2026**.

The model is fine-tuned from **`sarvamai/sarvam-translate`** using **QLoRA / LoRA with PEFT**.

## Model

* **Task:** English → Kashmiri translation
* **Base model:** `sarvamai/sarvam-translate`
* **Base architecture:** Gemma 3 4B IT
* **Fine-tuning:** QLoRA / LoRA
* **Framework:** PEFT
* **Inference:** BF16
* **Year:** 2026

### Model Weights

The trained adapter weights are hosted on Hugging Face:

**`KatheBathe/Kathe-Bathe`**

The inference script loads the base model and the KatheBathe adapter from the Hugging Face repository.

---

# Methodology

The model was developed using parameter-efficient fine-tuning.

```text
Sarvam-Translate
      │
      ▼
Gemma 3 4B IT
      │
      ▼
QLoRA / LoRA Fine-tuning
      │
      ▼
KatheBathe Adapter
      │
      ▼
English → Kashmiri Translation
```

QLoRA was used to fine-tune the model while keeping the base model frozen and training a relatively small number of additional parameters.

The resulting adapter is loaded on top of the original `sarvamai/sarvam-translate` model during inference.

---

# Training Data

The model was fine-tuned using:

### Kashmiri-English Parallel Corpus

**`SMUQamar/Kashmiri-English-Parallel-Corpus`**

https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Parallel-Corpus

### BPCC

**`ai4bharat/BPCC`**

https://huggingface.co/datasets/ai4bharat/BPCC

---

# Inference

The repository contains the inference implementation used to load the model and generate translations.

The script supports:

1. **Single-sentence inference**
2. **Batch CSV inference**
3. **Automatic Kaggle CSV detection**

---

# Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KATHE-KatheBathe
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install \
    "transformers==4.51.3" \
    "peft==0.15.2" \
    "accelerate" \
    "sentencepiece" \
    "safetensors" \
    "pandas"
```

---

# Model Loading

The inference script loads:

```text
Base Model
    ↓
sarvamai/sarvam-translate
    ↓
KatheBathe QLoRA Adapter
    ↓
Ready for inference
```

The adapter weights are loaded from the submitted Hugging Face repository.

A merged model is not required.

---

# Single-Sentence Inference

Example:

```bash
python inference.py \
    --text "She was a true visionary."
```

The script loads the base model and KatheBathe adapter and generates the Kashmiri translation.

Example:

```text
Input:
She was a true visionary.

Output:
سۄ ٲس اکھ حقیقی بصیرت تھون واجیٚنۍ۔
```

---

# Batch Inference

The script accepts a CSV containing:

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

---

# Kaggle Usage

Enable a GPU in Kaggle:

```text
Notebook
→ Settings
→ Accelerator
→ GPU
```

Then run:

```bash
python inference.py
```

The script can automatically search:

```text
/kaggle/input/**/*.csv
```

for compatible input files.

If multiple CSV files are available, an input file can be specified manually:

```bash
python inference.py \
    --input /kaggle/input/my-dataset/test.csv \
    --output /kaggle/working/predictions.csv
```

---

# Generation Configuration

The inference configuration used for the submitted model is:

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

Generation uses deterministic decoding:

```python
do_sample=False
```

If GPU memory is insufficient, the batch size can be reduced:

```bash
python inference.py --batch-size 8
```

---

# Inference Pipeline

```text
Input
  │
  ├── Single sentence
  │
  └── CSV batch
        │
        ▼
Load tokenizer
        │
        ▼
Load Sarvam-Translate
        │
        ▼
Load KatheBathe QLoRA adapter
        │
        ▼
Generate Kashmiri translation
        │
        ▼
Validate predictions
        │
        ▼
Save output CSV
```

For batch inference, the script validates the generated output, including:

* Prediction count
* Empty predictions
* ID order
* Output columns
* Saved CSV row count
* Saved CSV IDs

---

# Repository Structure

```text
KATHE-KatheBathe/
│
├── inference.py
├── README.md
└── requirements.txt
```

The trained model weights are hosted separately on Hugging Face.

---

# Requirements

The main software dependencies are:

```text
Python
PyTorch
Transformers 4.51.3
PEFT 0.15.2
Accelerate
SentencePiece
Safetensors
Pandas
```

BF16 inference requires a compatible NVIDIA GPU.

---

# Reproducibility

The submitted inference code is intended to load the same model adapter uploaded to Hugging Face.

The complete inference setup therefore consists of:

```text
GitHub
    │
    └── Inference code
            │
            ▼
Hugging Face
    │
    ├── KatheBathe adapter
    └── Tokenizer/configuration
            │
            ▼
sarvamai/sarvam-translate
            │
            ▼
English → Kashmiri
```

This allows the submitted code to be tested against the submitted model weights.

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

# Acknowledgements

We acknowledge the creators of:

* `sarvamai/sarvam-translate`
* `SMUQamar/Kashmiri-English-Parallel-Corpus`
* `ai4bharat/BPCC`
* Gemma 3

Their models and datasets were important resources in developing KatheBathe.

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
