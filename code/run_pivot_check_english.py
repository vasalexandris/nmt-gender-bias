#!/usr/bin/env python3
"""Robustness check of the English pivot route with an open model (EuroLLM-9B-Instruct Q4_K_M, greedy decoding).
For each of the 112 original sentences: source -> English, then English -> target (the original target language).
Output: data_pivot_check/{ES_EL,EL_ES}_english_route_eurollm.csv. Compare with the Claude-generated English route in the *_deepl_nextgen.csv files.
Usage (repo root): python code/run_pivot_check_english.py <path-to-gguf>"""
import csv, sys
from llama_cpp import Llama
SYS = "You are a professional translator. Respond with only the translation, no explanations or notes."
llm = Llama(model_path=sys.argv[1], n_ctx=512, n_threads=4, verbose=False)
def tr(text, s, t):
    u = f"Translate the following text from {s} into {t}.\n{s}: {text}\n{t}:"
    o = llm.create_chat_completion(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": u}], temperature=0.0, max_tokens=128)
    return o['choices'][0]['message']['content'].strip()
for d, f, col, s, t in (('ES_EL', 'ES_EL_deepl_nextgen.csv', 'Source_Spanish', 'Spanish', 'Greek'), ('EL_ES', 'EL_ES_deepl_nextgen.csv', 'Source_Greek', 'Greek', 'Spanish')):
    rows = list(csv.DictReader(open(f, encoding='utf-8-sig'))); out = []
    for r in rows:
        en = tr(r[col], s, 'English'); bt = tr(en, 'English', t)
        out.append([r['ID'], r[col], en, bt, r['Pivot_English_Translation'], r[f'Pivot_English_BackTrans_{t}']]); print(d, r['ID'], en, '|', bt, flush=True)
    w = csv.writer(open(f'data_pivot_check/{d}_english_route_eurollm.csv', 'w', encoding='utf-8-sig', newline=''))
    w.writerow(['ID', 'Source', 'English_EuroLLM', 'BackTrans_EuroLLM', 'English_Claude', 'BackTrans_Claude']); w.writerows(out)
