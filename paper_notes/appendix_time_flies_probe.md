# Appendix draft: the "time flies" probe (exploratory; not part of the 112 sentences)

Status: exploratory, one run per system, outputs in `data_pivot_check/time_flies_probe.csv` (script `code/probe_time_flies.py`). Open systems only (NLLB-200 distilled 1.3B, EuroLLM-9B-Instruct Q4_K_M, OPUS-MT). Google Translate and DeepL were not run on the verb idiom.

## What the dataset items are
EL->ES IDs 40 and 41 use the Greek noun phrases «Μύγες του χρόνου» and «Μύγες χρόνου» ("flies of time"; gold «Las moscas del año», «Moscas de tiempo»). They are **not** the Greek idiom «Ο χρόνος πετά». In English the noun phrase and the idiom collapse into the same string, *time flies*. A system that passes through English can therefore read the noun as a verb.

Observed on the dataset items (annotation: noun phrase is correct):
- Google Translate and DeepL Classic: «El tiempo vuela» (the verb idiom) on both items — marked as errors.
- DeepL Next Gen, NLLB, EuroLLM: «(Las) moscas del tiempo» — noun kept.
- The English route of EuroLLM (a separate probe in `data_pivot_check/`) is «Time flies» for both, i.e. the bridge string is the ambiguous one.

## Probe on the verb idiom and the noun phrase, open systems
| Direction | Source | NLLB | EuroLLM | OPUS-MT direct | OPUS-MT via English |
|---|---|---|---|---|---|
| EL->ES | Ο χρόνος πετά. (idiom) | El tiempo vuela. | El tiempo vuela. | — | Time's flying. -> El tiempo vuela. |
| EL->ES | Μύγες του χρόνου (noun) | Las moscas del tiempo | Moscas del tiempo | — | Time fluctuations -> Fluctuaciones del tiempo |
| ES->EL | El tiempo vuela. (idiom) | Ο χρόνος πετάει. | Ο χρόνος πετάει. | Ο χρόνος περνάει. | Time flies. -> Ο χρόνος περνάει. |
| ES->EL | Moscas de tiempo (noun) | Μύγες του καιρού | Μύγες του χρόνου | Μύγες του χρόνου | **Time flies. -> Ο χρόνος πετάει** (noun read as verb) |
| FR->PL | Le temps vole. (idiom) | Czas ucieka. | Czas leci. | Czas leci. | Time flies. -> Czas leci. |
| FR->PL | Mouches du temps (noun) | Węże czasu (wrong noun) | Muchy czasu | Machy czasu (garbled) | Weaknesses of time -> Słabości czasu (wrong) |
| PL->FR | Czas leci. (idiom) | Le temps passe. | Le temps passe. | Le temps passe. | Time flies. -> Le temps s'envole. |
| PL->FR | Muchy czasu (noun) | Les mouches du temps | Les mouches du temps | Les mouches du temps | Flies of time -> Les mouches du temps |

## Reading
- The verb idiom is translated correctly by every open system in every direction: the trap does not work in the reverse (verb -> noun) direction.
- The noun phrase -> verb error is reproduced by the English-pivoted OPUS-MT system in ES->EL («Moscas de tiempo» -> «Time flies» -> «Ο χρόνος πετάει»), as by Google and DeepL Classic in EL->ES. The open end-to-end systems (NLLB, EuroLLM) keep the noun.
- The English route of EuroLLM yields the ambiguous string «Time flies» for the noun phrases in all four cases, which is the mechanism the English-pivot hypothesis predicts; the final EuroLLM output does not follow it because the system translates directly.
- Limits: a single run; two noun-phrase items in the dataset; the OPUS-MT noun-phrase errors in FR->PL are lexical failures of small models, not the English trap.
