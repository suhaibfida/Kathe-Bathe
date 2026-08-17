#!/usr/bin/env python
"""
KATHE / KatheBathe
English -> Kashmiri inference using the Hugging Face QLoRA adapter.

INPUT MODES
============

1. Automatic Kaggle CSV:
       python inference.py

   Searches /kaggle/input/**/*.csv and automatically selects the
   only CSV containing both:
       ID
       sentence

   The CSV filename can be anything.

2. Specific CSV:
       python inference.py \
           --input /kaggle/input/my-dataset/myfile.csv \
           --output /kaggle/working/predictions.csv

3. Custom examples:
       python inference.py \
           --text "She was a true visionary." \
           --text "The weather is beautiful today."

OUTPUT
======

CSV columns:
    ID,kashmiri_text

MODEL
=====

Uses BF16 inference, matching the proven KATHE scoring setup.

Only ONE diagnostic sentence is tested before full inference.
No 10-row sanity test is performed.
"""

import argparse
import glob
import os

import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "sarvamai/sarvam-translate"
ADAPTER = "KatheBathe/Kathe-Bathe"

MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 232

NUM_BEAMS = 6
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3

BATCH_SIZE = 16

SYSTEM_PROMPT = (
    "Translate the text below to Kashmiri. "
    "Return only the translation."
)

DEFAULT_OUTPUT = (
    "/kaggle/working/"
    "submission_sarvam_kashmiri_bf16.csv"
)

REQUIRED_COLUMNS = {"ID", "sentence"}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "KatheBathe English -> Kashmiri "
            "BF16 inference."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input CSV containing ID and sentence. "
            "If omitted, automatically searches "
            "/kaggle/input/**/*.csv."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size. Default: 16.",
    )

    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help=(
            "Directly translate a custom sentence. "
            "Can be supplied multiple times."
        ),
    )

    return parser.parse_args()


# ============================================================
# AUTOMATIC CSV DISCOVERY
# ============================================================

def find_input_csv(explicit_path=None):

    if explicit_path:

        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Input file not found:\n{explicit_path}"
            )

        return explicit_path


    if not os.path.isdir("/kaggle/input"):
        raise FileNotFoundError(
            "No --input was provided and /kaggle/input "
            "does not exist. Use --input PATH or --text."
        )


    print("\n" + "=" * 70)
    print("AUTOMATIC CSV DISCOVERY")
    print("=" * 70)


    csv_files = glob.glob(
        "/kaggle/input/**/*.csv",
        recursive=True,
    )


    print("CSV files found:", len(csv_files))


    if not csv_files:
        raise FileNotFoundError(
            "No CSV files were found under /kaggle/input/."
        )


    candidates = []


    for path in csv_files:

        try:

            preview = pd.read_csv(
                path,
                nrows=5,
            )

            columns = set(preview.columns)


            if REQUIRED_COLUMNS.issubset(columns):

                candidates.append(path)

                print("Compatible:", path)

            else:

                print(
                    "Skipped:",
                    path,
                    "| columns:",
                    list(preview.columns),
                )


        except Exception as exc:

            print(
                "Skipped:",
                path,
                "| read error:",
                str(exc),
            )


    if not candidates:
        raise FileNotFoundError(
            "\nNo compatible CSV was found.\n\n"
            "The CSV must contain:\n"
            "    ID\n"
            "    sentence\n\n"
            "Or use --input PATH."
        )


    if len(candidates) > 1:
        raise RuntimeError(
            "\nMultiple compatible CSV files were found:\n\n"
            + "\n".join(
                f"  {path}"
                for path in candidates
            )
            + "\n\n"
            "The script will not guess.\n"
            "Use --input PATH to select the correct file."
        )


    selected = candidates[0]

    print("\nSelected input:", selected)

    return selected


# ============================================================
# DIRECT TEXT INPUT
# ============================================================

def create_text_dataframe(texts):

    if not texts:
        return None


    return pd.DataFrame(
        {
            "ID": range(1, len(texts) + 1),
            "sentence": texts,
        }
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(df):

    required_columns = ["ID", "sentence"]


    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}. "
                f"Expected columns: {required_columns}"
            )


    if df["sentence"].isna().any():
        raise ValueError(
            "Input CSV contains NULL sentences."
        )


    if (
        df["sentence"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Input CSV contains empty sentences."
        )


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():

    print("\n" + "=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)


    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER,
        trust_remote_code=True,
    )


    tokenizer.padding_side = "left"


    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False


    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer has no pad_token_id."
        )


    print("Tokenizer loaded.")
    print("Vocab size:", len(tokenizer))


    return tokenizer


# ============================================================
# BASE MODEL + QLORA ADAPTER
# ============================================================

def load_model():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. "
            "This BF16 inference script requires "
            "an NVIDIA GPU."
        )


    print("\n" + "=" * 70)
    print("LOADING BASE MODEL")
    print("=" * 70)


    print("Base model:", BASE_MODEL)


    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


    base_model.eval()


    print("Base model loaded.")


    print("\n" + "=" * 70)
    print("LOADING QLORA ADAPTER")
    print("=" * 70)


    print("Adapter:", ADAPTER)


    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
    )


    model.eval()


    print("QLoRA adapter loaded.")
    print("COMPLETE MODEL LOADED FROM HUGGING FACE")


    return model


# ============================================================
# OUTPUT CLEANING
# ============================================================

def clean_output(decoded):

    for line in decoded.splitlines():

        line = line.strip()


        if not line:
            continue


        if line.startswith("<unused"):
            continue


        if line in {
            "<pad>",
            "<eos>",
            "</s>",
        }:
            continue


        return line


    return ""


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(
    model,
    tokenizer,
    sources,
):

    sources = [
        str(x).strip()
        for x in sources
    ]


    prompts = []


    for source in sources:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": source,
            },
        ]


        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


        prompts.append(prompt)


    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
        add_special_tokens=False,
    )


    input_device = next(
        model.parameters()
    ).device


    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # NaN / Inf diagnostic
    # --------------------------------------------------------

    with torch.no_grad():

        check = model(
            **inputs,
            use_cache=False,
        )


        logits = check.logits


    if not torch.isfinite(logits).all():

        bad_count = (
            ~torch.isfinite(logits)
        ).sum().item()


        raise RuntimeError(
            "NaN/Inf logits detected. "
            f"Bad values: {bad_count}"
        )


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    prompt_length = (
        inputs["input_ids"].shape[1]
    )


    results = []


    for i in range(len(sources)):

        generated_tokens = outputs[i][
            prompt_length:
        ]


        decoded = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )


        results.append(
            clean_output(decoded)
        )


    return results


# ============================================================
# ONE-SENTENCE TEST ONLY
# ============================================================

def run_one_sentence_test(
    model,
    tokenizer,
):

    print("\n" + "=" * 70)
    print("ONE-SENTENCE TEST")
    print("=" * 70)


    sentence = "She was a true visionary."


    result = translate_batch(
        model,
        tokenizer,
        [sentence],
    )[0]


    print("\nEnglish:")
    print(sentence)


    print("\nKashmiri:")
    print(repr(result))


    if not result.strip():
        raise RuntimeError(
            "STOP: one-sentence test "
            "produced empty output."
        )


    print("\nONE-SENTENCE TEST PASSED")


# ============================================================
# FULL TRANSLATION
# ============================================================

def run_full_translation(
    model,
    tokenizer,
    df,
    batch_size,
):

    sources = (
        df["sentence"]
        .astype(str)
        .tolist()
    )


    predictions = []


    total = len(sources)


    print("\n" + "=" * 70)
    print("FULL TRANSLATION")
    print("=" * 70)


    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )


        batch = sources[start:end]


        batch_predictions = translate_batch(
            model,
            tokenizer,
            batch,
        )


        if len(batch_predictions) != len(batch):

            raise RuntimeError(
                f"Batch mismatch: {start}:{end}"
            )


        predictions.extend(
            batch_predictions
        )


        print(
            f"{len(predictions):,}/{total:,}"
        )


    return predictions


# ============================================================
# SUBMISSION
# ============================================================

def create_submission(
    df,
    predictions,
):

    if len(predictions) != len(df):

        raise RuntimeError(
            f"Expected {len(df)} predictions, "
            f"got {len(predictions)}."
        )


    empty_count = sum(
        not str(x).strip()
        for x in predictions
    )


    print("\nPrediction check:")
    print("Predictions:", len(predictions))
    print("Empty predictions:", empty_count)


    if empty_count > 0:

        raise RuntimeError(
            "Empty predictions found. Stopping."
        )


    submission = pd.DataFrame(
        {
            "ID": df["ID"],
            "kashmiri_text": predictions,
        }
    )


    if (
        submission["ID"].tolist()
        != df["ID"].tolist()
    ):
        raise RuntimeError(
            "ID order changed."
        )


    if list(submission.columns) != [
        "ID",
        "kashmiri_text",
    ]:
        raise RuntimeError(
            "Unexpected output columns."
        )


    return submission


# ============================================================
# SAVE + VERIFY
# ============================================================

def save_and_verify(
    submission,
    output_path,
):

    output_dir = os.path.dirname(
        os.path.abspath(output_path)
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    submission.to_csv(
        output_path,
        index=False,
    )


    check = pd.read_csv(
        output_path
    )


    if len(check) != len(submission):

        raise RuntimeError(
            "Saved CSV has the wrong number of rows."
        )


    if list(check.columns) != [
        "ID",
        "kashmiri_text",
    ]:

        raise RuntimeError(
            "Saved CSV has incorrect columns."
        )


    if (
        check["ID"].tolist()
        != submission["ID"].tolist()
    ):

        raise RuntimeError(
            "Saved CSV IDs do not match input IDs."
        )


    if (
        check["kashmiri_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Saved CSV contains empty translations."
        )


    print("\nSaved and verified:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be at least 1."
        )


    if args.input and args.text:

        raise ValueError(
            "Use either --input or --text, not both."
        )


    # --------------------------------------------------------
    # Select input
    # --------------------------------------------------------

    if args.text:

        test_df = create_text_dataframe(
            args.text
        )

        input_description = "Direct custom text"


    else:

        input_path = find_input_csv(
            args.input
        )


        print(
            "\nInput selected:",
            input_path,
        )


        test_df = pd.read_csv(
            input_path
        )


        input_description = input_path


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)


    print("Input:", input_description)
    print("Rows:", len(test_df))
    print("Columns:", list(test_df.columns))


    validate_input(test_df)


    print("Input validation: PASSED")


    # --------------------------------------------------------
    # Load tokenizer and model
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    model = load_model()


    # --------------------------------------------------------
    # ONLY ONE diagnostic sentence
    # --------------------------------------------------------

    run_one_sentence_test(
        model,
        tokenizer,
    )


    # --------------------------------------------------------
    # Full inference
    # --------------------------------------------------------

    predictions = run_full_translation(
        model,
        tokenizer,
        test_df,
        args.batch_size,
    )


    # --------------------------------------------------------
    # Create submission
    # --------------------------------------------------------

    submission = create_submission(
        test_df,
        predictions,
    )


    print("\n" + "=" * 70)
    print("SUBMISSION VALIDATION PASSED")
    print("=" * 70)


    save_and_verify(
        submission,
        args.output,
    )


    print("\n" + "=" * 70)
    print("FIRST 10 RESULTS")
    print("=" * 70)


    for i in range(
        min(10, len(submission))
    ):

        print("\n" + "-" * 70)

        print(
            "ID:",
            submission.iloc[i]["ID"],
        )

        print(
            "English:",
            test_df.iloc[i]["sentence"],
        )

        print(
            "Kashmiri:",
            submission.iloc[i]["kashmiri_text"],
        )


    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()