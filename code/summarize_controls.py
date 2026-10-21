#!/usr/bin/env python3
"""Summarize the OPUS-MT control systems (data_controls_opus/) against the closed and open-weight systems.

Controls (all Marian bilingual transformers, 6+6 layers, trained from scratch per pair):
  Direct bilingual   es->el                      (no direct el->es / el->en model of usable quality exists, see README)
  Cascade via English  es->en->el ; el->en->es
  Cascade via French   es->fr->el ; el->fr->es   (gender-preserving pivot control)
Gender_Bias for the controls was assigned with the same rule as the other systems (1 = gender lost/changed, or
tense/number/person change, or lost polysemy sense; purely lexical = 0). Judgments are in code/opus_control_judgments.py.
"""
import csv, sys
sys.path.insert(0, 'code')
from opus_control_judgments import ES_EL, EL_ES, DIAG_ES, DIAG_EL

dd = [r for r in csv.DictReader(open('data_pivot_diagnostic/gender_marker_diagnostic_subset.csv', encoding='utf-8-sig')) if r['Diagnostic'] == 'YES']
S = lambda col: sum(int(float(r[col])) for r in dd)

print("Bias rate (Gender_Bias=1), n=56 per direction")
print(f"{'':20s} {'ES->EL':>10s} {'EL->ES':>10s}")
bias = {
    'Direct bilingual': (sum(v[0] for v in ES_EL.values()), None),
    'Cascade via English': (sum(v[1] for v in ES_EL.values()), sum(v[0] for v in EL_ES.values())),
    'Cascade via French': (sum(v[2] for v in ES_EL.values()), sum(v[1] for v in EL_ES.values())),
}
for k, (a, b) in bias.items():
    print(f"{k:20s} {a:4d} ({a/56:5.1%}) " + (f"{b:4d} ({b/56:5.1%})" if b is not None else "      n/a"))

print("\nGender-marker diagnostic subset (match = output carries the English route's biased marker)")
es = list(zip(*DIAG_ES.values())); el = list(zip(*DIAG_EL.values()))
rows = {'Cascade via English': (sum(es[1]) + sum(el[0]), 15), 'Google': (S('Google_Match'), 15),
        'DeepL Classic': (S('DeepLClassic_Match'), 15), 'Cascade via French': (sum(es[2]) + sum(el[1]), 15),
        'NLLB-1.3B': (S('NLLB_Match'), 15), 'Direct bilingual (ES->EL only)': (sum(es[0]), 13),
        'EuroLLM-9B': (S('EuroLLM_Match'), 15), 'DeepL Next Gen': (S('DeepLNextGen_Match'), 15)}
for k, (m, n) in sorted(rows.items(), key=lambda x: -x[1][0] / x[1][1]):
    print(f"  {k:32s} {m:2d}/{n} = {m/n:5.1%}")
