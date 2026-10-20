import json, subprocess, sys, pandas as pd, torch
from transformers import MarianMTModel, MarianTokenizer
def sha(m): return json.loads(subprocess.run(['curl','-sS','--max-time','30',f'https://huggingface.co/api/models/{m}'],capture_output=True,text=True).stdout)['sha']
class MT:
    def __init__(self, name):
        self.rev = sha(name); self.name = name
        self.tok = MarianTokenizer.from_pretrained(name, revision=self.rev); self.m = MarianMTModel.from_pretrained(name, revision=self.rev).eval()
        print('loaded', name, self.rev, flush=True)
    def __call__(self, text):
        enc = self.tok([text], return_tensors='pt', truncation=True)
        with torch.no_grad(): g = self.m.generate(**enc, num_beams=5, max_length=200)
        return self.tok.batch_decode(g, skip_special_tokens=True)[0]
out = sys.argv[1]
es_fr, fr_el = MT('Helsinki-NLP/opus-mt-es-fr'), MT('Helsinki-NLP/opus-mt-fr-el')
df = pd.read_csv('data_nllb/ES_EL_nllb.csv'); rows = []
for _, r in df.iterrows():
    mid = es_fr(r.Source_Spanish); rows.append(dict(ID=r.ID, Source_Spanish=r.Source_Spanish, French_pivot=mid, Cascade_es_fr_el=fr_el(mid)))
pd.DataFrame(rows).to_csv(out + 'opus_es_el_french.csv', index=False); print('ES-EL french done', flush=True)
el_en, en_es = MT('Helsinki-NLP/opus-mt-ine-en'), MT('Helsinki-NLP/opus-mt-en-es')
el_fr, fr_es = MT('Helsinki-NLP/opus-mt-el-fr'), MT('Helsinki-NLP/opus-mt-fr-es')
df = pd.read_csv('data_nllb/EL_ES_nllb.csv'); rows = []
for _, r in df.iterrows():
    s = r.Source_Greek; e = el_en(s); f = el_fr(s)
    rows.append(dict(ID=r.ID, Source_Greek=s, English_pivot=e, Cascade_el_en_es=en_es(e), French_pivot=f, Cascade_el_fr_es=fr_es(f)))
pd.DataFrame(rows).to_csv(out + 'opus_el_es_both.csv', index=False); print('DONE', flush=True)
