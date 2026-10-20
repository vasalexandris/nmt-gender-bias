#!/usr/bin/env python3
"""FR<->PL replication: bias rates, English-biased-subset rates and McNemar tests (run from repo root).
E_* = sentences (non-N) where the English-route back-translation is itself biased vs gold (judged by the annotator, protocol Section 6)."""
import csv, math, collections
E_FR = {2,3,4,5,6,7,8,16,18,24,25,27,28,29,32,33,34,35,38,40,41,42,43,44,45,46,47,48,51,55}
E_PL = {1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,22,23,24,26,27,28,29,32,33,35,36,37,38,39,50,53}
def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1); n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10; k = min(n01, n10)
    return n10, n01, (min(1, 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else 1)
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d, (c+h)/d
for d, E in (('FR_PL', E_FR), ('PL_FR', E_PL)):
    rows = list(csv.DictReader(open(f'data_fr_pl/{d}_annotations.csv', encoding='utf-8-sig')))
    by = collections.defaultdict(dict)
    for r in rows:
        if r['Signal_In_Source'] != 'N': by[r['System']][int(r['ID'])] = int(r['Gender_Bias'])
    ids = sorted(by['NLLB']); print(f"\n{d}: scored n={len(ids)} (N sentences excluded); English-route-biased subset n={len(E & set(ids))}")
    for s in ['OPUS_EN', 'OPUS_ES', 'OPUS_direct', 'NLLB', 'EuroLLM']:
        k = sum(by[s][i] for i in ids); n = len(ids); lo, hi = wilson(k, n)
        ke = sum(by[s][i] for i in ids if i in E); ne = len(E & set(ids)); le, he = wilson(ke, ne)
        print(f"  {s:12s} bias {k}/{n} = {k/n:.1%} [{lo:.0%}-{hi:.0%}] | English-biased subset {ke}/{ne} = {ke/ne:.1%} [{le:.0%}-{he:.0%}]")
    for a, b in (('OPUS_EN', 'OPUS_direct'), ('OPUS_EN', 'OPUS_ES'), ('OPUS_direct', 'OPUS_ES'), ('NLLB', 'OPUS_direct'), ('EuroLLM', 'OPUS_direct')):
        n10, n01, p = mcnemar([by[a][i] for i in ids], [by[b][i] for i in ids]); print(f"  McNemar exact {a} vs {b}: only-{a}={n10}, only-{b}={n01}, p={p:.4f}")
