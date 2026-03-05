# The English Trap: Gender Bias and Grammatical Information Loss Through English-Influenced Universal Representations in Multilingual NMT

**Vasileios Alexandris & Nikolaos Asimopoulos**  
Department of Electrical and Computer Engineering, University of Western Macedonia, Kozani, Greece

---

## Overview

This repository contains the dataset and code accompanying the paper:

> *The English Trap: Gender Bias and Grammatical Information Loss Through English-Influenced Universal Representations in Multilingual NMT*

We challenge the "language-agnostic interlingua" claim of multilingual NMT systems, demonstrating that Transformer-based models (Google Translate, DeepL Classic) develop **English-influenced representations** that systematically discard grammatical gender, while LLM-based systems (DeepL Next Gen) largely escape this bias.

---

## Dataset

**112 annotated sentences** across 2 translation directions:
- **Spanish → Greek** (56 sentences)
- **Greek → Spanish** (56 sentences)

Evaluated on **3 MT systems**:
| System | Architecture | Gender Error Rate |
|--------|-------------|-------------------|
| Google Translate | Transformer | 75.9% |
| DeepL Classic | Transformer | 70.5% |
| DeepL Next Gen | LLM-based | 6.25% |

Pivot language detection across **16 typologically diverse languages**:
English, Swedish, Turkish, French, German, Russian, Polish, Hungarian, Italian, Albanian, Chinese, Japanese, Arabic, Hebrew, Hindi, Swahili

---

## Repository Structure

```
├── complete_dataset_v6.xlsx          # Full dataset (all systems, both directions)
├── EL_ES_google_translate.csv        # Greek→Spanish, Google Translate
├── EL_ES_deepl_classic.csv           # Greek→Spanish, DeepL Classic
├── EL_ES_deepl_nextgen.csv           # Greek→Spanish, DeepL Next Gen
├── ES_EL_google_translate.csv        # Spanish→Greek, Google Translate
├── ES_EL_deepl_classic.csv           # Spanish→Greek, DeepL Classic
├── ES_EL_deepl_nextgen.csv           # Spanish→Greek, DeepL Next Gen
└── statistical_analysis.py           # Full statistical analysis script
```

---

## CSV Format

Each CSV file contains **56 rows × 63 columns**:

| Columns | Description |
|---------|-------------|
| `ID` | Sentence identifier (1–56) |
| `Source_Spanish` / `Source_Greek` | Source sentence |
| `Gender_Bias_Focus` | Target gender feature |
| `Category` | Error category (Personal Pronouns, Adjectives, etc.) |
| `Gold_Greek` / `Gold_Spanish` | Reference translation |
| `MT_Output_Greek` / `MT_Output_Spanish` | MT system output |
| `Gender_Bias` | 1 = gender error, 0 = correct |
| `Justification` | Human annotation explaining the error |
| `Error_*` (7 columns) | Error type flags (gender generalization, tense change, etc.) |
| `Pivot_X_Translation` (×16) | Source sentence translated to pivot language X |
| `Pivot_X_BackTrans_*` (×16) | Back-translation from pivot X to target language |
| `Pivot_X_Score` (×16) | 1 = exact match with MT output, 0 = no match |

---

## Methodology

We use a **black-box pivot language detection** approach:

1. Translate each source sentence to candidate language *L* (using Claude Sonnet 4.5, temperature=0, with strict gender-preservation constraints)
2. Translate from *L* to the target language
3. Count exact matches with the commercial MT system's direct output
4. Apply softmax normalization over all 16 candidates to obtain pivot probabilities

A high probability for English indicates that the system's internal representation behaves as if routed through English.

---

## Key Results

| System | English Pivot Probability | Gender Accuracy | Cohen's h vs Next Gen |
|--------|--------------------------|-----------------|----------------------|
| Google Translate | **97.3%** | 24.1% | 1.61 (Huge) |
| DeepL Classic | 61.2% (p=0.35, n.s.) | 29.5% | 1.49 (Huge) |
| DeepL Next Gen | <0.1% | **93.75%** | — |

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@dataset{alexandris_2026_englishtrap,
  author    = {Alexandris, Vasileios and Asimopoulos, Nikolaos},
  title     = {The English Trap: Gender Bias and Grammatical Information Loss 
               Through English-Influenced Universal Representations in Multilingual NMT},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
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
