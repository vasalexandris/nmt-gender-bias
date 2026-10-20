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
es_en, en_el = MT('Helsinki-NLP/opus-mt-es-en'), MT('Helsinki-NLP/opus-mt-en-el')
es_el = MT('Helsinki-NLP/opus-mt-es-el')
df = pd.read_csv('data_nllb/ES_EL_nllb.csv'); rows = []
for _, r in df.iterrows():
    s = r.Source_Spanish; eng = es_en(s)
    rows.append(dict(ID=r.ID, Source_Spanish=s, Gold_Greek=r.Gold_Greek, Direct_es_el=es_el(s), Cascade_English=eng, Cascade_es_en_el=en_el(eng)))
    print(r.ID, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1] + 'opus_es_el.csv', index=False)
el_en, en_es = MT('Helsinki-NLP/opus-mt-tc-big-el-en'), MT('Helsinki-NLP/opus-mt-en-es')
df = pd.read_csv('data_nllb/EL_ES_nllb.csv'); rows = []
for _, r in df.iterrows():
    s = r.Source_Greek; eng = el_en(s)
    rows.append(dict(ID=r.ID, Source_Greek=s, Gold_Spanish=r.Gold_Spanish, Cascade_English=eng, Cascade_el_en_es=en_es(eng)))
    print('el', r.ID, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1] + 'opus_el_es.csv', index=False)
print('DONE', flush=True)
