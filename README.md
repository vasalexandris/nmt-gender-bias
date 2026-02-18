# English-Influenced Interlingua in Multilingual NMT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18686006.svg)](https://doi.org/10.5281/zenodo.18686006)
[![Paper](https://img.shields.io/badge/Paper-Computational%20Linguistics-blue)](https://github.com/vasalexandris/nmt-gender-bias)

This repository contains the dataset, code, and supplementary materials for the paper:

> **The English Trap: Gender Bias and Grammatical Information Loss Through English-Influenced Universal Representations in Multilingual NMT**
>
> Vasileios Alexandris, Nikolaos Asimopoulos
>
> *Submitted to Computational Linguistics, 2026*

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🔍 Overview

This study challenges the "language-agnostic" representation claim in multilingual neural machine translation (NMT) systems. Using a novel **black-box pivot detection methodology**, we demonstrate that Transformer-based models develop **English-influenced interlinguae** that systematically discard grammatical gender.

### Main Contributions

1. **Novel Methodology**: Black-box pivot language detection using softmax probabilities over 16 typologically diverse languages
2. **Empirical Evidence**: Quantitative proof of overwhelming English dominance (97.3% in Google Translate)
3. **Error Taxonomy**: Mechanistic mapping between English morphological gaps and gender bias patterns
4. **Architectural Solution**: Demonstration that LLM-based systems escape the "English Trap" (10-fold bias reduction)

## 🎯 Key Findings

### Gender Bias Rates

| System | Errors | Total | Bias Rate |
|--------|--------|-------|-----------|
| Google Translate | 85 | 112 | **75.9%** |
| DeepL Classic | 79 | 112 | **70.5%** |
| DeepL Next Gen | 7 | 112 | **6.25%** ✅ |

### Pivot Language Dominance

**Google Translate:**
- English: **97.3%** probability
- All other 15 languages: 2.7%

**DeepL Next Gen:**
- No single dominant pivot
- English: **<0.1%** probability
- Balanced distribution across morphologically rich languages

### Effect Size

- Google → DeepL Next Gen: **Cohen's h = 1.57** (huge effect)
- Bias reduction: **10.7×** improvement

## 📊 Dataset

### Test Set Specifications

- **Total sentences**: 112 (56 per direction)
- **Language pair**: Modern Greek (EL) ↔ European Spanish (ES)
- **Focus**: Gender agreement phenomena
- **Categories**: Pronouns, adjectives, demonstratives, professional titles, etc.

### Selection Criteria

✅ At least one gender-agreement trigger  
✅ Unambiguous semantic context  
✅ Balanced feminine/masculine subjects (~50/50)  
✅ Controlled length (8-15 words)  
✅ Common vocabulary and simple syntax  

### Candidate Pivot Languages (16)

| Gender System | Languages |
|---------------|-----------|
| No gender | English, Turkish, Hungarian, Chinese, Japanese, Swahili |
| Two genders | French, Italian, Hebrew, Arabic, Hindi |
| Three genders | German, Russian, Polish |
| Hybrid/other | Swedish, Albanian |

## 🛠️ Installation

### Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Dependencies

- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `openpyxl>=3.0.9` (for Excel file handling)

## 🚀 Usage

### Reproduce Statistical Analysis

```bash
python code/statistical_analysis.py
```

This script computes:
- Softmax probabilities from matching scores
- Cohen's h effect sizes
- Summary statistics

### Load Dataset

```python
import pandas as pd

# Load Greek→Spanish dataset
df_el_es = pd.read_csv('data/greek_spanish_dataset.csv')

# Load Spanish→Greek dataset
df_es_el = pd.read_csv('data/spanish_greek_dataset.csv')

# Load pivot scores
pivot_scores = pd.read_csv('data/pivot_scores.csv')
```

### Example: Analyze Pivot Scores

```python
import pandas as pd

# Load pivot scores
df = pd.read_csv('data/pivot_scores.csv')

# Filter for Greek→Spanish direction
df_el_es = df[df['Direction'] == 'Greek→Spanish']

# Find English dominance
english_scores = df_el_es[df_el_es['Language'] == 'English']
print(english_scores)
```

## 📁 Repository Structure

```
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── complete_dataset.xlsx   # Full dataset (3 sheets)
│   ├── greek_spanish_dataset.csv
│   ├── spanish_greek_dataset.csv
│   └── pivot_scores.csv
│
├── code/
│   └── statistical_analysis.py # Reproduce all calculations
│
└── paper/
    └── alexandris_asimopoulos_2026.pdf  # (Added after publication)
```

## 📄 Dataset Files

### `complete_dataset.xlsx`

Excel workbook with 3 sheets:
- **Spanish**: Spanish→Greek direction (56 sentences)
- **Scores**: Pivot language matching scores
- **Greek**: Greek→Spanish direction (56 sentences)

### `greek_spanish_dataset.csv`

Columns:
- `ID`: Sentence identifier (1-56)
- `Source_Greek`: Source sentence in Greek
- `Focus`: Gender agreement trigger element
- `Category`: Error category
- `Gold_Spanish`: Gold reference translation
- `Google_Output`: Google Translate output
- `Google_Error_Description`: Error description
- `Google_Bias`: Binary bias flag (1=error, 0=correct)

### `pivot_scores.csv`

Columns:
- `Direction`: Translation direction
- `Language`: Candidate pivot language
- `Google_Matches`: Number of exact matches (out of 56)
- `DeepL_Classic_Matches`: Number of exact matches
- `DeepL_NextGen_Matches`: Number of exact matches

## 🔬 Methodology

### Pivot Detection Algorithm

For each sentence and candidate language L:

1. **Step 1**: Translate source → L using Claude Sonnet 4.5 with gender-neutral constraint
2. **Step 2**: Translate L → target naturally
3. **Step 3**: Compare with commercial system's direct output (exact match)
4. **Scoring**: Count exact matches across all 56 sentences
5. **Softmax**: Convert raw scores to normalized probabilities

```python
P(L | system) = exp(score_L) / Σ exp(score_k)
```

### Quality Controls

- ✅ Native-speaker verification on 20% sample
- ✅ Consistency check via 10% re-simulation
- ✅ Symmetry verification across directions

## 📖 Citation

If you use this dataset or methodology, please cite:

```bibtex
@article{alexandris2026englishtrap,
  title={The English Trap: Gender Bias and Grammatical Information Loss Through
         English-Influenced Universal Representations in Multilingual NMT},
  author={Alexandris, Vasileios and Asimopoulos, Nikolaos},
  journal={Under Review},
  year={2026},
  note={Under review}
}
```

Or use the `CITATION.cff` file for automatic citation generation.

## 📜 License

This dataset is released under the [MIT License](LICENSE).

You are free to:
- ✅ Use the data for academic research
- ✅ Modify and distribute the code
- ✅ Use in commercial applications (with attribution)

## 🤝 Contributing

We welcome contributions! If you find any issues or have suggestions:

1. Open an issue
2. Submit a pull request
3. Contact us directly (see below)

## 👥 Contact

**Vasileios Alexandris**  
PhD Candidate  
Department of Electrical and Computer Engineering  
University of Western Macedonia, Greece  
📧 vasalexandris76@gmail.com

**Nikolaos Asimopoulos**  
Professor  
Department of Electrical and Computer Engineering  
University of Western Macedonia, Greece  
📧 nasimop@uowm.gr

## 🙏 Acknowledgments

We thank Anthropic for providing access to Claude Sonnet 4.5 for the pivot detection simulations.

## 📚 Related Work

- [WinoMT](https://github.com/gabrielstanovsky/mt_gender) - Gender bias evaluation benchmark
- [MuST-SHE](https://ict.fbk.eu/must-she/) - Gender translation benchmark
- [Google's Multilingual NMT](https://arxiv.org/abs/1611.04558) - Original zero-shot translation paper

---

**Star ⭐ this repository if you find it useful!**

Last updated: February 2026
