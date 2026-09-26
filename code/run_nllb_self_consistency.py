#!/usr/bin/env python3
"""Self-consistency pivot test for NLLB: instead of comparing NLLB's direct
output against Claude-produced back-translations (style mismatch confound),
re-translate each already-existing Pivot_{lang}_Translation leg-1 text BACK
to the target language using NLLB itself, then compare to NLLB's own direct
output. Same translator on both routes -> style differences can't cause
false non-matches.

Result / caveat (see project discussion): this removes the style-mismatch
confound but introduces a different one -- NLLB's own per-language-pair
translation quality varies a lot (much better for e.g. French/German than
Hindi/Chinese), so a "match" on this test can reflect general translation
quality for that pivot language rather than genuine internal pivot routing.
Verified concretely on the ID 38 "bates" polysemy case: NLLB's own French and
Hindi legs produced the wrong ("bats" the animal) sense too, even though the
French/Hindi leg-1 text is unambiguous -- i.e. NLLB's own translation errors
on those legs, not a shared ambiguity, caused the false-positive match. For
this reason the paper's primary evidence uses the Claude-based back-translation
method (fixed, high-quality leg-2 for all systems: code/compute_pivot_scores.py
and the "uniquely-diagnostic subset" analysis), with this self-consistency
script kept as a documented robustness check, not the primary metric.
"""
import sys
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MODEL_REVISION = "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"

PIVOT_LANG_CODES = {
    'English': 'eng_Latn', 'Swedish': 'swe_Latn', 'Turkish': 'tur_Latn',
    'French': 'fra_Latn', 'German': 'deu_Latn', 'Russian': 'rus_Cyrl',
    'Polish': 'pol_Latn', 'Hungarian': 'hun_Latn', 'Italian': 'ita_Latn',
    'Albanian': 'als_Latn', 'Chinese': 'zho_Hans', 'Japanese': 'jpn_Jpan',
    'Arabic': 'arb_Arab', 'Hebrew': 'heb_Hebr', 'Hindi': 'hin_Deva',
    'Swahili': 'swh_Latn',
}
TARGET_CODES = {'Greek': 'ell_Grek', 'Spanish': 'spa_Latn'}

REPO = "."


def load_model():
    print(f"Loading {MODEL_NAME}@{MODEL_REVISION} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()
    return tok, model


def translate_batch(tok, model, sentences, src_lang, tgt_lang, batch_size=8):
    tok.src_lang = src_lang
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    outputs = []
    for i in range(0, len(sentences), batch_size):
        batch = [str(s) for s in sentences[i:i + batch_size]]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=200, num_beams=5)
        outputs.extend(tok.batch_decode(generated, skip_special_tokens=True))
    return outputs


def run_direction(tok, model, nllb_csv, target_lang_name, out_path):
    df = pd.read_csv(nllb_csv)
    result = {"ID": df["ID"], "NLLB_Output": df["NLLB_Output"]}
    tgt_code = TARGET_CODES[target_lang_name]

    for lang, src_code in PIVOT_LANG_CODES.items():
        leg1_col = f"Pivot_{lang}_Translation"
        leg1_texts = df[leg1_col].tolist()
        selfconsistent_out = translate_batch(tok, model, leg1_texts, src_code, tgt_code)
        result[f"NLLB_SelfBackTrans_{lang}"] = selfconsistent_out
        result[f"NLLB_SelfScore_{lang}"] = [
            int(a == b) for a, b in zip(selfconsistent_out, df["NLLB_Output"])
        ]
        matches = sum(result[f"NLLB_SelfScore_{lang}"])
        print(f"  {lang:10s} ({src_code} -> {tgt_code}): {matches}/{len(df)} match NLLB's direct output", file=sys.stderr)

    out_df = pd.DataFrame(result)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    tok, model = load_model()

    print("\n=== ES->EL (target: Greek) ===", file=sys.stderr)
    run_direction(tok, model, f"{REPO}/data_nllb/ES_EL_nllb.csv", "Greek",
                  f"{REPO}/data_nllb/ES_EL_nllb_selfconsistency.csv")

    print("\n=== EL->ES (target: Spanish) ===", file=sys.stderr)
    run_direction(tok, model, f"{REPO}/data_nllb/EL_ES_nllb.csv", "Spanish",
                  f"{REPO}/data_nllb/EL_ES_nllb_selfconsistency.csv")


if __name__ == "__main__":
    main()
