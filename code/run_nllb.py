#!/usr/bin/env python3
"""Translate the 112-sentence dataset with NLLB-200 (open-weight) for both directions.

Reproducibility note: this pins the exact HuggingFace Hub commit of the model
checkpoint used in the paper's validation experiment, so re-running this script
against the same commit reproduces the same weights regardless of any later
updates to the "main" branch of the model repository.

Model: facebook/nllb-200-distilled-1.3B
Pinned revision (HF Hub commit hash): 7be3e24664b38ce1cac29b8aeed6911aa0cf0576
Environment used for the paper's run: Python 3.11, torch==2.14.0+cpu, transformers==5.17.0

Usage:
    python run_nllb.py
Requires: pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install transformers sentencepiece pandas
Outputs (relative to repo root): data_nllb/ES_EL_nllb.csv, data_nllb/EL_ES_nllb.csv
"""

import sys
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MODEL_REVISION = "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
LANG_CODES = {"spa": "spa_Latn", "ell": "ell_Grek"}

REPO = "."


def load_model():
    print(f"Loading {MODEL_NAME}@{MODEL_REVISION} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()
    return tokenizer, model


def translate_batch(tokenizer, model, sentences, src_lang, tgt_lang, batch_size=8):
    tokenizer.src_lang = src_lang
    outputs = []
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=200,
            num_beams=5,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)
        print(f"  translated {min(i + batch_size, len(sentences))}/{len(sentences)}", file=sys.stderr)
    return outputs


def run_direction(tokenizer, model, csv_path, source_col, gold_col, src_lang, tgt_lang, out_path):
    df = pd.read_csv(csv_path)
    sentences = df[source_col].tolist()
    nllb_out = translate_batch(tokenizer, model, sentences, src_lang, tgt_lang)

    result = pd.DataFrame({
        "ID": df["ID"],
        source_col: df[source_col],
        "Gender_Bias_Focus": df["Gender_Bias_Focus"],
        "Category": df["Category"],
        gold_col: df[gold_col],
        "NLLB_Output": nllb_out,
        "Gender_Bias": "",       # to be annotated manually, same criteria as the other systems
        "Justification": "",
    })
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    tokenizer, model = load_model()

    run_direction(
        tokenizer, model,
        csv_path=f"{REPO}/ES_EL_google_translate.csv",
        source_col="Source_Spanish", gold_col="Gold_Greek",
        src_lang=LANG_CODES["spa"], tgt_lang=LANG_CODES["ell"],
        out_path=f"{REPO}/data_nllb/ES_EL_nllb.csv",
    )

    run_direction(
        tokenizer, model,
        csv_path=f"{REPO}/EL_ES_google_translate.csv",
        source_col="Source_Greek", gold_col="Gold_Spanish",
        src_lang=LANG_CODES["ell"], tgt_lang=LANG_CODES["spa"],
        out_path=f"{REPO}/data_nllb/EL_ES_nllb.csv",
    )


if __name__ == "__main__":
    main()
