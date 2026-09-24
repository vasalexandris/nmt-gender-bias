#!/usr/bin/env python3
"""Translate the 112-sentence dataset with EuroLLM-9B-Instruct (open-weight, GGUF).

Reproducibility note: EuroLLM-9B-Instruct's official weights (utter-project/EuroLLM-9B-Instruct
on HuggingFace) are gated behind a license click-through, which would make third-party
verification depend on someone's HF account approval. To keep this fully third-party
reproducible, we instead use a public, non-gated GGUF re-quantization of the exact same
base model, pinned to a specific commit, and run it locally with llama.cpp (CPU-only,
deterministic greedy decoding).

Model: bartowski/EuroLLM-9B-Instruct-GGUF, file EuroLLM-9B-Instruct-Q4_K_M.gguf (Q4_K_M quantization)
Base model: utter-project/EuroLLM-9B-Instruct (Apache-2.0)
Pinned revision (HF Hub commit hash of the GGUF repo): 55e5590cc457e42ed197c34c5b10403472b05d54
Environment used for the paper's run: llama-cpp-python==0.3.35, CPU inference, n_ctx=512

Usage:
    python run_eurollm.py <path-to-downloaded-gguf-file>

To fetch the pinned file yourself:
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="bartowski/EuroLLM-9B-Instruct-GGUF",
        filename="EuroLLM-9B-Instruct-Q4_K_M.gguf",
        revision="55e5590cc457e42ed197c34c5b10403472b05d54",
    )

Outputs (relative to repo root): data_eurollm/ES_EL_eurollm.csv, data_eurollm/EL_ES_eurollm.csv
"""

import sys
import pandas as pd
from llama_cpp import Llama

GGUF_REVISION = "55e5590cc457e42ed197c34c5b10403472b05d54"  # bartowski/EuroLLM-9B-Instruct-GGUF commit
REPO = "."

SYSTEM_PROMPT = "You are a professional translator. Respond with only the translation, no explanations or notes."


def load_model(gguf_path):
    print(f"Loading {gguf_path} (pinned to commit {GGUF_REVISION}) ...", file=sys.stderr)
    return Llama(model_path=gguf_path, n_ctx=512, n_threads=4, verbose=False)


def translate(llm, text, src_lang, tgt_lang):
    user_content = f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {text}\n{tgt_lang}:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,   # greedy / deterministic decoding
        max_tokens=128,
    )
    return out["choices"][0]["message"]["content"].strip()


def run_direction(llm, csv_path, source_col, gold_col, src_lang, tgt_lang, out_path):
    df = pd.read_csv(csv_path)
    outputs = []
    for i, text in enumerate(df[source_col].tolist()):
        translated = translate(llm, text, src_lang, tgt_lang)
        outputs.append(translated)
        print(f"  [{src_lang}->{tgt_lang}] {i+1}/{len(df)}: {text!r} -> {translated!r}", file=sys.stderr)

    result = pd.DataFrame({
        "ID": df["ID"],
        source_col: df[source_col],
        "Gender_Bias_Focus": df["Gender_Bias_Focus"],
        "Category": df["Category"],
        gold_col: df[gold_col],
        "EuroLLM_Output": outputs,
        "Gender_Bias": "",       # to be annotated manually, same criteria as the other systems
        "Justification": "",
    })
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eurollm.py <path-to-gguf-file>", file=sys.stderr)
        sys.exit(1)
    llm = load_model(sys.argv[1])

    run_direction(
        llm,
        csv_path=f"{REPO}/ES_EL_google_translate.csv",
        source_col="Source_Spanish", gold_col="Gold_Greek",
        src_lang="Spanish", tgt_lang="Greek",
        out_path=f"{REPO}/data_eurollm/ES_EL_eurollm.csv",
    )

    run_direction(
        llm,
        csv_path=f"{REPO}/EL_ES_google_translate.csv",
        source_col="Source_Greek", gold_col="Gold_Spanish",
        src_lang="Greek", tgt_lang="Spanish",
        out_path=f"{REPO}/data_eurollm/EL_ES_eurollm.csv",
    )


if __name__ == "__main__":
    main()
