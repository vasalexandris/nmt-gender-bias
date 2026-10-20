#!/usr/bin/env python3
"""Run the open systems on the FR<->PL source sentences (protocol/PROTOCOL_FR_PL_replication.md).
Usage (repo root): python code/run_fr_pl.py {nllb|eurollm <gguf>|opus}
Reads data_fr_pl/source_sentences_as_is.csv; writes data_fr_pl/outputs/<DIR>_<system>.csv. Settings identical to run_nllb.py / run_eurollm.py /
run_opus_controls_part*.py; the OPUS-MT revisions are the ones pinned in the protocol."""
import csv, json, subprocess, sys
SRC = 'data_fr_pl/source_sentences_as_is.csv'
DIRS = {'FR_PL': ('French', 'Polish', 'fra_Latn', 'pol_Latn'), 'PL_FR': ('Polish', 'French', 'pol_Latn', 'fra_Latn')}
rows = {d: [r for r in csv.DictReader(open(SRC, encoding='utf-8-sig')) if r['Direction'] == d] for d in DIRS}

def save(d, name, cols):
    out = f'data_fr_pl/outputs/{d}_{name}.csv'
    base = rows[d]; keys = ['ID', 'Source_As_Is'] + list(cols)
    w = csv.writer(open(out, 'w', encoding='utf-8-sig', newline='')); w.writerow(keys)
    for i, r in enumerate(base): w.writerow([r['Original_ID'], r['Source_As_Is']] + [cols[c][i] for c in cols])
    print('wrote', out, flush=True)

def nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name, rev = 'facebook/nllb-200-distilled-1.3B', '7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
    tok = AutoTokenizer.from_pretrained(name, revision=rev); m = AutoModelForSeq2SeqLM.from_pretrained(name, revision=rev).eval()
    for d, (_, _, sl, tl) in DIRS.items():
        tok.src_lang = sl; sents = [r['Source_As_Is'] for r in rows[d]]; out = []
        for i in range(0, len(sents), 8):
            enc = tok(sents[i:i+8], return_tensors='pt', padding=True, truncation=True)
            g = m.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tl), max_length=200, num_beams=5)
            out += tok.batch_decode(g, skip_special_tokens=True)
        save(d, 'nllb', {'NLLB_Output': out})

def eurollm(path):
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=512, n_threads=4, verbose=False)
    sysmsg = "You are a professional translator. Respond with only the translation, no explanations or notes."
    for d, (sl, tl, _, _) in DIRS.items():
        out = []
        for r in rows[d]:
            u = f"Translate the following text from {sl} into {tl}.\n{sl}: {r['Source_As_Is']}\n{tl}:"
            o = llm.create_chat_completion(messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": u}], temperature=0.0, max_tokens=128)
            out.append(o['choices'][0]['message']['content'].strip()); print(d, r['Original_ID'], out[-1], flush=True)
        save(d, 'eurollm', {'EuroLLM_Output': out})

def opus():
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    class MT:
        def __init__(s, n):
            rev = json.loads(subprocess.run(['curl', '-sS', '--max-time', '30', f'https://huggingface.co/api/models/{n}'], capture_output=True, text=True).stdout)['sha']
            s.t = MarianTokenizer.from_pretrained(n, revision=rev); s.m = MarianMTModel.from_pretrained(n, revision=rev).eval(); print('loaded', n, rev, flush=True)
        def __call__(s, x, pre=''):
            e = s.t([pre + x], return_tensors='pt', truncation=True)
            with torch.no_grad(): g = s.m.generate(**e, num_beams=5, max_length=200)
            return s.t.batch_decode(g, skip_special_tokens=True)[0]
    H = 'Helsinki-NLP/opus-mt-'
    pairs = {'FR_PL': dict(direct=[(H+'fr-pl', '')], en=[(H+'fr-en', ''), (H+'en-zlw', '>>pol<< ')], es=[(H+'fr-es', ''), (H+'es-pl', '')]),
             'PL_FR': dict(direct=[(H+'pl-fr', '')], en=[(H+'pl-en', ''), (H+'en-fr', '')], es=[(H+'pl-es', ''), (H+'es-fr', '')])}
    cache = {}
    for d, spec in pairs.items():
        cols = {}
        for key, hops in spec.items():
            ms = [(cache.setdefault(n, MT(n)), pre) for n, pre in hops]; outs = []; mids = []
            for r in rows[d]:
                x = r['Source_As_Is']
                for k, (mdl, pre) in enumerate(ms):
                    x = mdl(x, pre)
                    if k == 0 and len(ms) > 1: mids.append(x)
                outs.append(x)
            if mids: cols[f'{key}_Pivot'] = mids
            cols[f'{key}_Output'] = outs
        save(d, 'opus', cols)

{'nllb': nllb, 'eurollm': lambda: eurollm(sys.argv[2]), 'opus': opus}[sys.argv[1]]()
