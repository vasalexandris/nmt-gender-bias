#!/usr/bin/env python3
"""Exploratory probe (not part of the 112 sentences): the idiom 'time flies' (verb) versus the noun phrase 'flies of time'.
Runs the open systems on the verb idiom in both language pairs and on the existing noun phrases for reference.
Usage (repo root): python code/probe_time_flies.py <path-to-gguf>; writes data_pivot_check/time_flies_probe.csv"""
import csv, json, subprocess, sys, torch
ITEMS = [  # (direction, source, label)
 ('EL_ES', 'Ο χρόνος πετά.', 'verb idiom'), ('EL_ES', 'Μύγες του χρόνου', 'noun phrase (dataset ID40)'), ('EL_ES', 'Μύγες χρόνου', 'noun phrase (dataset ID41)'),
 ('ES_EL', 'El tiempo vuela.', 'verb idiom'), ('ES_EL', 'Moscas de tiempo', 'noun phrase'),
 ('FR_PL', 'Le temps vole.', 'verb idiom'), ('FR_PL', 'Mouches du temps', 'noun phrase'),
 ('PL_FR', 'Czas leci.', 'verb idiom'), ('PL_FR', 'Muchy czasu', 'noun phrase')]
L = {'EL': ('Greek', 'ell_Grek'), 'ES': ('Spanish', 'spa_Latn'), 'FR': ('French', 'fra_Latn'), 'PL': ('Polish', 'pol_Latn')}
res = {k: {} for k in range(len(ITEMS))}
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
n, rv = 'facebook/nllb-200-distilled-1.3B', '7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
tok = AutoTokenizer.from_pretrained(n, revision=rv); m = AutoModelForSeq2SeqLM.from_pretrained(n, revision=rv).eval()
for k, (d, s, _) in enumerate(ITEMS):
    a, b = d.split('_'); tok.src_lang = L[a][1]
    with torch.no_grad(): g = m.generate(**tok([s], return_tensors='pt'), forced_bos_token_id=tok.convert_tokens_to_ids(L[b][1]), max_length=100, num_beams=5)
    res[k]['NLLB'] = tok.batch_decode(g, skip_special_tokens=True)[0]
del m
from llama_cpp import Llama
llm = Llama(model_path=sys.argv[1], n_ctx=512, n_threads=4, verbose=False)
for k, (d, s, _) in enumerate(ITEMS):
    a, b = d.split('_'); sl, tl = L[a][0], L[b][0]
    o = llm.create_chat_completion(messages=[{"role": "system", "content": "You are a professional translator. Respond with only the translation, no explanations or notes."}, {"role": "user", "content": f"Translate the following text from {sl} into {tl}.\n{sl}: {s}\n{tl}:"}], temperature=0.0, max_tokens=64)
    res[k]['EuroLLM'] = o['choices'][0]['message']['content'].strip()
    o = llm.create_chat_completion(messages=[{"role": "system", "content": "You are a professional translator. Respond with only the translation, no explanations or notes."}, {"role": "user", "content": f"Translate the following text from {sl} into English.\n{sl}: {s}\nEnglish:"}], temperature=0.0, max_tokens=64)
    res[k]['EuroLLM_English_route'] = o['choices'][0]['message']['content'].strip()
def sha(x): return json.loads(subprocess.run(['curl', '-sS', '--max-time', '30', f'https://huggingface.co/api/models/{x}'], capture_output=True, text=True).stdout)['sha']
class MT:
    def __init__(s, name, pre=''):
        r = sha(name); s.t = MarianTokenizer.from_pretrained(name, revision=r); s.m = MarianMTModel.from_pretrained(name, revision=r).eval(); s.pre = pre
    def __call__(s, x):
        with torch.no_grad(): g = s.m.generate(**s.t([s.pre + x], return_tensors='pt'), num_beams=5, max_length=100)
        return s.t.batch_decode(g, skip_special_tokens=True)[0]
H = 'Helsinki-NLP/opus-mt-'
routes = {'ES_EL': [MT(H+'es-en'), MT(H+'en-el')], 'EL_ES': [MT(H+'ine-en'), MT(H+'en-es')], 'FR_PL': [MT(H+'fr-en'), MT(H+'en-zlw', '>>pol<< ')], 'PL_FR': [MT(H+'pl-en'), MT(H+'en-fr')]}
direct = {'ES_EL': MT(H+'es-el'), 'FR_PL': MT(H+'fr-pl'), 'PL_FR': MT(H+'pl-fr')}
for k, (d, s, _) in enumerate(ITEMS):
    e = routes[d][0](s); res[k]['OPUS_English_route'] = e; res[k]['OPUS_via_English'] = routes[d][1](e)
    if d in direct: res[k]['OPUS_direct'] = direct[d](s)
cols = ['NLLB', 'EuroLLM', 'EuroLLM_English_route', 'OPUS_direct', 'OPUS_English_route', 'OPUS_via_English']
with open('data_pivot_check/time_flies_probe.csv', 'w', encoding='utf-8-sig', newline='') as h:
    w = csv.writer(h); w.writerow(['Direction', 'Source', 'Type'] + cols)
    for k, (d, s, t) in enumerate(ITEMS): w.writerow([d, s, t] + [res[k].get(c, '') for c in cols]); print(d, s, [res[k].get(c, '') for c in cols], flush=True)
