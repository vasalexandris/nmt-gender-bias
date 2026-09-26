#!/usr/bin/env python3
"""Summarize the gender-marker diagnostic subset (data_pivot_diagnostic/gender_marker_diagnostic_subset.csv).

Methodology (three refinements applied in sequence, each addressing a specific
weakness found in discussion while validating the open-weight models):

1. Raw aggregate softmax over all 56/112 sentences (original method) -- diluted
   by sentences where most/all of the 16 pivot languages happen to agree.
2. "Uniquely-diagnostic subset": restrict to the 24 sentences where English's
   pivot back-translation differs from all other 15 candidate languages --
   removes the dilution, but still counts sentences where English's route
   happens to be the CORRECT answer, or where the divergence is about number/
   formality rather than grammatical gender (both would misattribute evidence).
3. "Gender-marker diagnostic subset" (this script): further restricts to the
   15 sentences where (a) English's route is uniquely different from all other
   pivots AND (b) English's route itself diverges from Gold specifically on
   the tested gender feature. Within that set, a "match" means the system's
   output carries the same gender/number marker as English's route (not
   necessarily an identical string) -- e.g. Google's "τιμές" vs NLLB's
   "άριστα" in ID6 both drop the required feminine pronoun, so both count as
   matches despite different wording.

Each row's Diagnostic/Exclusion_Reason/*_Match value was assigned by manual
linguistic review (documented per-row in the CSV), not by automated string
matching, since determining "the gender-bearing token" requires reading each
sentence's morphology.
"""
import pandas as pd

REPO = "."
CSV_PATH = f"{REPO}/data_pivot_diagnostic/gender_marker_diagnostic_subset.csv"

SYSTEMS = {
    'Google Translate': 'Google',
    'DeepL Classic': 'DeepLClassic',
    'DeepL Next Gen': 'DeepLNextGen',
    'NLLB (open)': 'NLLB',
    'EuroLLM (open)': 'EuroLLM',
}


def main():
    df = pd.read_csv(CSV_PATH)
    diag = df[df['Diagnostic'] == 'YES']
    excluded = df[df['Diagnostic'] == 'NO']

    print("=" * 80)
    print("GENDER-MARKER DIAGNOSTIC SUBSET SUMMARY")
    print("=" * 80)
    print(f"\nUnique-backtrans sentences (English differs from all 15 other pivots): {len(df)}")
    print(f"Genuinely gender-bias-diagnostic (English route also diverges from Gold "
          f"on the tested gender feature): {len(diag)}")
    print(f"Excluded: {len(excluded)}")
    for _, row in excluded.iterrows():
        print(f"  {row['Direction']} ID{row['ID']}: {row['Exclusion_Reason']}")

    print(f"\n{'System':18s} {'Matches':>10s} {'Rate':>8s}")
    print("-" * 40)
    for label, col_prefix in SYSTEMS.items():
        m = diag[f'{col_prefix}_Match'].sum()
        n = len(diag)
        print(f"{label:18s} {int(m):3d}/{n:<3d}   {m/n:6.1%}")


if __name__ == "__main__":
    main()
