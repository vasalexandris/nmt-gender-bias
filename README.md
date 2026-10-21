# Is English the Hidden Pivot? A Black-Box Study of Gender, Number and Lexical Information Loss in Greek–Spanish Machine Translation

**Vasileios Alexandris & Nikolaos Asimopoulos**  
Department of Electrical and Computer Engineering, University of Western Macedonia, Kozani, Greece

> **Status: revision in progress.** The title and framing changed (previous title: *The English Trap: Gender Bias and Grammatical Information Loss Through English-Influenced Universal Representations in Multilingual NMT*). Numbers below are working values and are locked at the release that receives the final DOI. The citation block still shows the DOI of an earlier version.

---

## Overview

This repository contains the data and code for a **behavioural (black-box)** study of whether machine-translation systems lose information in a way that is consistent with routing through English. It does **not** probe internal representations; it is complementary to work that analyses the latent language of open-weight models (e.g. Kudugunta et al., 2019; Wendler et al., 2024; Schut et al., 2025; Zhong et al., 2025).

Design principle: the 112 sentences (56 Spanish→Greek, 56 Greek→Spanish) are **English-lossy by design**. Each carries a feature that English drops or merges: the grammatical gender of third-person plural forms, articles and participles; the number and formality of *you* (*yours*, *yourself*); and lexically ambiguous words (*bat*, *seal*, ...). They are targeted, not representative: reported rates are failure rates on such structures, not general bias rates.

What the evidence supports, and what it does not:
- **Gender** loss alone cannot identify English as the pivot: other languages without grammatical gender (Swedish, Turkish, Hungarian, ...) yield the same masculine default in the black-box comparison.
- **English-specific features** (number/formality of *you*, polysemy) separate English from the other 15 candidate pivot languages; see `data_pivot_diagnostic/english_specific_subset.csv`.
- Controls with bilingual and English-pivoted OPUS-MT systems show that the pattern depends on the translation regime, not on the Transformer architecture alone (`data_controls_opus/`).

---

## Systems

| System | Type | Reproducible? | Gender error rate (112 sentences) |
|--------|------|---------------|-------------------|
| Google Translate | commercial NMT | no (dated snapshot) | 75.9% |
| DeepL Classic | commercial NMT (announced as replaced by Next Gen) | no (dated snapshot) | 69.6% |
| DeepL Next Gen | commercial, LLM-based | no (dated snapshot) | 6.25% |
| NLLB-200 distilled 1.3B | open encoder–decoder NMT | yes (pinned revision) | 43.8% |
| EuroLLM-9B-Instruct (GGUF Q4_K_M) | open decoder-only LLM | yes (pinned revision) | 25.0% (draft annotation) |

The closed systems were queried by the authors on dates that were not recorded; the outputs released here are the analysed snapshot. Pivot translations and back-translations were generated with Claude Sonnet 4.5 (temperature 0, instruction to preserve gender strictly, two steps); the exact prompts were not archived. The generated texts themselves are released, so all analyses can be re-run from the data.

Pivot detection uses **16 candidate languages**: English, Swedish, Turkish, French, German, Russian, Polish, Hungarian, Italian, Albanian, Chinese, Japanese, Arabic, Hebrew, Hindi, Swahili.

---

## Repository Structure

```
├── complete_dataset_v6.xlsx / complete_dataset_v9.xlsx   # Full dataset (v9 adds open-model and control blocks)
├── {ES_EL,EL_ES}_{google_translate,deepl_classic,deepl_nextgen}.csv   # closed systems
├── data_nllb/  data_eurollm/                # open-weight system outputs, annotations, pivot scores
├── data_controls_opus/                      # OPUS-MT controls (direct, English cascade, French cascade)
├── data_pivot_diagnostic/                   # gender-marker and English-specific diagnostic subsets
├── data_fr_pl/                              # second language pair (French<->Polish), open systems only
├── protocol/                                # pre-specified protocol for the second pair (frozen 2026-10-20)
├── code/                                    # scripts: running open systems, annotations, analyses
├── paper_notes/                             # drafts: limitations, related work, availability statement
└── requirements*.txt
```

---

## Open systems and controls (reproduction)

- NLLB-200 distilled 1.3B: `facebook/nllb-200-distilled-1.3B@7be3e24664b38ce1cac29b8aeed6911aa0cf0576`, beam 5 (`code/run_nllb.py`).
- EuroLLM-9B-Instruct: `bartowski/EuroLLM-9B-Instruct-GGUF@55e5590cc457e42ed197c34c5b10403472b05d54`, Q4_K_M, greedy decoding, llama.cpp (`code/run_eurollm.py`).
- OPUS-MT bilingual and cascade controls with pinned revisions: `code/run_opus_controls_part*.py`, `data_controls_opus/README.md`.
- Second language pair (French<->Polish): `protocol/PROTOCOL_FR_PL_replication.md`, `code/run_fr_pl.py`, `code/analyze_fr_pl.py`.

---

## CSV Format

Each closed-system CSV has **56 rows x 63 columns**: `ID`, source sentence, `Gender_Bias_Focus`, `Category`, gold translation, system output, `Gender_Bias` (1 = error: gender lost or changed, or tense/number/person changed, or polysemous sense lost; purely lexical differences = 0), `Justification`, error-type flags, and for each of the 16 pivot languages the pivot translation, its back-translation and an exact-match score.

---

## Citation

The DOI below belongs to an earlier version and is replaced when the final version is archived.

```bibtex
@dataset{alexandris_2026_hiddenpivot,
  author    = {Alexandris, Vasileios and Asimopoulos, Nikolaos},
  title     = {Is English the Hidden Pivot? A Black-Box Study of Gender, Number and Lexical
               Information Loss in Greek--Spanish Machine Translation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18881815},
  url       = {https://doi.org/10.5281/zenodo.18881815}
}
```

---

## License

This dataset is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
Copyright (C) 2026 The Authors.

---

## Contact

- Vasileios Alexandris — vasalexandris76@gmail.com  
- Nikolaos Asimopoulos — nasimop@uowm.gr
