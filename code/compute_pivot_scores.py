#!/usr/bin/env python3
"""Extend the open-weight model CSVs (data_nllb/, data_eurollm/) with the same
16-language pivot-detection columns used for the three original systems
(Google Translate, DeepL Classic, DeepL Next Gen).

The pivot translations and back-translations (Pivot_X_Translation,
Pivot_X_BackTrans_*) do not depend on which MT/LLM system is being scored --
they were computed once from the source sentences and are identical across
all system CSVs (verified: Google/DeepL Classic/DeepL Next Gen all carry the
exact same values). This script reuses that already-computed data and only
computes the new Pivot_X_Score columns: 1 if the open-weight model's output
is character-for-character identical to a given pivot's back-translation,
0 otherwise -- the exact same criterion already used for the three closed
systems (verified empirically: Pivot_X_Score==1 iff MT_Output == BackTrans).

Usage:
    python compute_pivot_scores.py
"""

import pandas as pd

PIVOTS = [
    "English", "Swedish", "Turkish", "French", "German", "Russian", "Polish",
    "Hungarian", "Italian", "Albanian", "Chinese", "Japanese", "Arabic",
    "Hebrew", "Hindi", "Swahili",
]

REPO = "."


def extend(reference_csv, system_csv, source_col, gold_col, back_lang, output_col, system_name, out_path):
    ref = pd.read_csv(reference_csv)
    sysdf = pd.read_csv(system_csv)

    data = {
        "ID": ref["ID"],
        source_col: ref[source_col],
        "Gender_Bias_Focus": ref["Gender_Bias_Focus"],
        "Category": ref["Category"],
        gold_col: ref[gold_col],
        output_col: sysdf[output_col],
        "Gender_Bias": sysdf["Gender_Bias"],
        "Justification": sysdf["Justification"],
    }

    for lang in PIVOTS:
        trans_col = f"Pivot_{lang}_Translation"
        back_col = f"Pivot_{lang}_BackTrans_{back_lang}"
        score_col = f"Pivot_{lang}_Score"
        data[trans_col] = ref[trans_col]
        data[back_col] = ref[back_col]
        data[score_col] = (ref[back_col] == sysdf[output_col]).astype(int)

    out = pd.DataFrame(data)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    total = {lang: out[f"Pivot_{lang}_Score"].sum() for lang in PIVOTS}
    print(f"  {system_name} pivot match counts (out of {len(out)}): {total}")


def main():
    # NLLB
    extend(
        reference_csv=f"{REPO}/ES_EL_google_translate.csv",
        system_csv=f"{REPO}/data_nllb/ES_EL_nllb.csv",
        source_col="Source_Spanish", gold_col="Gold_Greek", back_lang="Greek",
        output_col="NLLB_Output", system_name="NLLB (ES->EL)",
        out_path=f"{REPO}/data_nllb/ES_EL_nllb.csv",
    )
    extend(
        reference_csv=f"{REPO}/EL_ES_google_translate.csv",
        system_csv=f"{REPO}/data_nllb/EL_ES_nllb.csv",
        source_col="Source_Greek", gold_col="Gold_Spanish", back_lang="Spanish",
        output_col="NLLB_Output", system_name="NLLB (EL->ES)",
        out_path=f"{REPO}/data_nllb/EL_ES_nllb.csv",
    )

    # EuroLLM
    extend(
        reference_csv=f"{REPO}/ES_EL_google_translate.csv",
        system_csv=f"{REPO}/data_eurollm/ES_EL_eurollm.csv",
        source_col="Source_Spanish", gold_col="Gold_Greek", back_lang="Greek",
        output_col="EuroLLM_Output", system_name="EuroLLM (ES->EL)",
        out_path=f"{REPO}/data_eurollm/ES_EL_eurollm.csv",
    )
    extend(
        reference_csv=f"{REPO}/EL_ES_google_translate.csv",
        system_csv=f"{REPO}/data_eurollm/EL_ES_eurollm.csv",
        source_col="Source_Greek", gold_col="Gold_Spanish", back_lang="Spanish",
        output_col="EuroLLM_Output", system_name="EuroLLM (EL->ES)",
        out_path=f"{REPO}/data_eurollm/EL_ES_eurollm.csv",
    )


if __name__ == "__main__":
    main()
