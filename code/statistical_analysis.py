#!/usr/bin/env python3
"""
Statistical Analysis for Multilingual NMT Gender Bias Study

This script computes:
1. Softmax probabilities from matching scores
2. Chi-square goodness-of-fit tests (one-way)
3. Cohen's h effect sizes
4. Summary statistics

Note on correlation analysis: The monotonic relationship between English-matching
and bias rates is reported as a monotonic pattern (descriptive evidence only).
Formal significance testing is not appropriate with n=3 systems.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist
import json

# ============================================================================
# Data: Pivot Language Matching Scores
# ============================================================================

# Spanish → Greek direction (out of 56 sentences)
# Source: complete_dataset_v6.xlsx, sheet 'Scores', Spanish-Greek section
spanish_greek_scores = {
    'Google Translate': {
        'English': 29, 'Russian': 23, 'Swedish': 20, 'Turkish': 19,
        'Chinese': 18, 'Japanese': 18, 'Swahili': 18, 'Hungarian': 17,
        'Albanian': 15, 'French': 14, 'German': 13, 'Polish': 13,
        'Italian': 13, 'Arabic': 13, 'Hebrew': 13, 'Hindi': 13
    },
    'DeepL Classic': {
        'English': 22, 'Russian': 20, 'Swedish': 18, 'Turkish': 18,
        'Chinese': 18, 'Japanese': 18, 'Swahili': 18, 'Hungarian': 17,
        'French': 16, 'German': 13, 'Polish': 13, 'Italian': 13,
        'Albanian': 12, 'Arabic': 12, 'Hebrew': 12, 'Hindi': 12
    },
    'DeepL Next Gen': {
        'Polish': 24, 'Italian': 24, 'Arabic': 24, 'Hebrew': 24,
        'Hindi': 24, 'Albanian': 23, 'German': 20, 'French': 18,
        'Swedish': 11, 'Russian': 11, 'English': 10, 'Turkish': 8,
        'Hungarian': 8, 'Chinese': 8, 'Japanese': 8, 'Swahili': 8
    }
}

# Greek → Spanish direction (out of 56 sentences)
# Source: complete_dataset_v6.xlsx, sheet 'Scores', Greek-Spanish section
greek_spanish_scores = {
    'Google Translate': {
        'English': 29, 'Turkish': 24, 'Swedish': 23, 'Russian': 21,
        'Hungarian': 19, 'Swahili': 19, 'Chinese': 18, 'Japanese': 18,
        'Albanian': 16, 'Arabic': 15, 'Hebrew': 14, 'Hindi': 14,
        'German': 13, 'French': 12, 'Polish': 12, 'Italian': 11
    },
    'DeepL Classic': {
        'English': 20, 'Swedish': 17, 'Turkish': 17, 'Chinese': 17,
        'Japanese': 16, 'Russian': 15, 'Hungarian': 15, 'Swahili': 15,
        'Albanian': 14, 'Hindi': 13, 'German': 12, 'Polish': 12,
        'Arabic': 12, 'French': 11, 'Italian': 11, 'Hebrew': 11
    },
    'DeepL Next Gen': {
        'Hebrew': 24, 'Italian': 23, 'Arabic': 23, 'French': 22,
        'German': 21, 'Polish': 21, 'Albanian': 20, 'Hindi': 18,
        'Chinese': 17, 'Japanese': 14, 'Russian': 13, 'Turkish': 12,
        'Hungarian': 12, 'English': 11, 'Swedish': 11, 'Swahili': 11
    }
}

# Gender bias error counts
bias_errors = {
    'Spanish→Greek': {'Google': 45, 'DeepL Classic': 42, 'DeepL Next Gen': 4},
    'Greek→Spanish': {'Google': 40, 'DeepL Classic': 37, 'DeepL Next Gen': 3},
    'Combined': {'Google': 85, 'DeepL Classic': 79, 'DeepL Next Gen': 7}
}

total_sentences = {'per_direction': 56, 'combined': 112}


# ============================================================================
# Functions
# ============================================================================

def softmax(scores_dict, temperature=1.0):
    """
    Compute softmax probabilities from matching scores.
    
    Args:
        scores_dict: Dictionary mapping language names to matching scores
        temperature: Temperature parameter (default 1.0)
    
    Returns:
        Dictionary mapping language names to probabilities
    """
    scores = np.array(list(scores_dict.values()))
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    return {lang: prob for lang, prob in zip(scores_dict.keys(), probs)}


def cohens_h(p1, p2):
    """
    Compute Cohen's h effect size for two proportions.
    
    h = 2 * [arcsin(sqrt(p1)) - arcsin(sqrt(p2))]
    
    Interpretation:
    - |h| < 0.2: small
    - 0.2 ≤ |h| < 0.5: small to medium
    - 0.5 ≤ |h| < 0.8: medium to large
    - |h| ≥ 0.8: large/huge
    
    Args:
        p1: First proportion
        p2: Second proportion
    
    Returns:
        Cohen's h value
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


# ============================================================================
# Analysis
# ============================================================================

def chi_square_goodness_of_fit(scores_dict):
    """
    One-way chi-square goodness-of-fit test.
    H0: All candidate languages match equally well.
    H1: At least one language matches significantly more/less.

    Args:
        scores_dict: Dictionary mapping language names to combined match counts

    Returns:
        chi2_stat, p_value, df
    """
    observed = np.array(list(scores_dict.values()), dtype=float)
    n_langs = len(observed)
    expected = np.full(n_langs, observed.sum() / n_langs)
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = n_langs - 1
    p_value = 1 - chi2_dist.cdf(chi2_stat, df)
    return chi2_stat, p_value, df


# ============================================================================
# Analysis
# ============================================================================

def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS: Multilingual NMT Gender Bias Study")
    print("=" * 80)

    # 1. Softmax probabilities
    print("\n1. SOFTMAX PROBABILITIES")
    print("-" * 80)

    for direction_name, direction_scores in [
        ('Spanish→Greek', spanish_greek_scores),
        ('Greek→Spanish', greek_spanish_scores)
    ]:
        print(f"\n{direction_name}:")

        for system, scores in direction_scores.items():
            probs = softmax(scores)
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  {system}:")
            for lang, prob in sorted_probs[:5]:
                print(f"    {lang:12s}: {prob:6.1%} ({scores[lang]}/56 matches)")

            if 'English' in probs:
                print(f"    {'English':12s}: {probs['English']:6.1%} *** ENGLISH ***")

    # 2. Combined direction softmax
    print("\n\n2. COMBINED DIRECTION SOFTMAX (REPORTED IN PAPER)")
    print("-" * 80)

    for system in ['Google Translate', 'DeepL Classic', 'DeepL Next Gen']:
        all_scores = {}
        for lang, score in spanish_greek_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score
        for lang, score in greek_spanish_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score

        combined_probs = softmax(all_scores)
        sorted_probs = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{system}:")
        for lang, prob in sorted_probs[:5]:
            print(f"  {lang:12s}: {prob:6.1%}")
        if 'English' in combined_probs:
            print(f"  {'English':12s}: {combined_probs['English']:6.1%} *** REPORTED VALUE ***")

    # 3. Chi-square goodness-of-fit
    print("\n\n3. CHI-SQUARE GOODNESS-OF-FIT TESTS")
    print("-" * 80)
    print("H0: All 16 candidate languages match equally well")
    print("Critical value at alpha=0.05, df=15: chi2 = 25.00")
    print()

    for system in ['Google Translate', 'DeepL Classic', 'DeepL Next Gen']:
        all_scores = {}
        for lang, score in spanish_greek_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score
        for lang, score in greek_spanish_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score

        chi2_stat, p_val, df = chi_square_goodness_of_fit(all_scores)
        n_langs = len(all_scores)
        english_score = all_scores.get('English', 0)
        english_pct = english_score / 112 * 100

        print(f"{system}:")
        print(f"  Languages: {n_langs}, df={df}")
        print(f"  chi2 = {chi2_stat:.2f}, p = {p_val:.4f}")
        print(f"  English: {english_score}/112 = {english_pct:.1f}%")
        top_lang = max(all_scores, key=all_scores.get)
        print(f"  Top language: {top_lang} ({all_scores[top_lang]}/112 = {all_scores[top_lang]/112*100:.1f}%)")
        print()

    # 4. Cohen's h effect sizes
    print("\n4. COHEN'S h EFFECT SIZES")
    print("-" * 80)

    google_rate = bias_errors['Combined']['Google'] / total_sentences['combined']
    deepl_c_rate = bias_errors['Combined']['DeepL Classic'] / total_sentences['combined']
    deepl_ng_rate = bias_errors['Combined']['DeepL Next Gen'] / total_sentences['combined']

    print(f"\nBias rates (combined):")
    print(f"  Google Translate: {google_rate:.1%} ({bias_errors['Combined']['Google']}/112)")
    print(f"  DeepL Classic:    {deepl_c_rate:.1%} ({bias_errors['Combined']['DeepL Classic']}/112)")
    print(f"  DeepL Next Gen:   {deepl_ng_rate:.1%} ({bias_errors['Combined']['DeepL Next Gen']}/112)")

    h_google_ng = cohens_h(google_rate, deepl_ng_rate)
    h_deepl_c_ng = cohens_h(deepl_c_rate, deepl_ng_rate)
    h_google_deepl_c = cohens_h(google_rate, deepl_c_rate)

    print(f"\nEffect sizes (Cohen's h):")
    print(f"  Google → DeepL Next Gen:        h = {h_google_ng:5.3f} (huge)")
    print(f"  DeepL Classic → DeepL Next Gen: h = {h_deepl_c_ng:5.3f} (huge)")
    print(f"  Google ↔ DeepL Classic:         h = {h_google_deepl_c:5.3f} (negligible)")
    print(f"\n  Note: Values computed from raw CSV data (Next Gen: 4 ES->EL + 3 EL->ES = 7/112).")
    print(f"  Qualitative interpretation (huge/negligible) is unchanged across all comparisons.")

    print("\nInterpretation (Cohen, 1988):")
    print("  |h| < 0.2:        negligible")
    print("  0.2 ≤ |h| < 0.5:  small")
    print("  0.5 ≤ |h| < 0.8:  medium")
    print("  |h| ≥ 0.8:        large/huge")

    # 5. Bias reduction
    print("\n\n5. BIAS REDUCTION")
    print("-" * 80)

    reduction_google = google_rate / deepl_ng_rate
    reduction_deepl_c = deepl_c_rate / deepl_ng_rate

    print(f"\nGoogle → DeepL Next Gen:        {reduction_google:.1f}× reduction")
    print(f"DeepL Classic → DeepL Next Gen: {reduction_deepl_c:.1f}× reduction")

    # 6. Monotonic correspondence (replaces Pearson r)
    print("\n\n6. ENGLISH-MATCHING vs BIAS RATES (MONOTONIC CORRESPONDENCE)")
    print("-" * 80)
    print("\nSystem               English-Matching   Bias Rate")
    print("Google Translate     51.8%              75.9%")
    print("DeepL Classic        37.5%              70.5%")
    print("DeepL Next Gen       18.8%               6.25%")
    print("Note: With n=3 systems, formal significance testing is not appropriate.")
    print("The pattern is perfectly monotonic across all 3 systems.")
    print("Note: DeepL Classic (chi2=16.47, p=0.35): English is #1 (37.5%) but not")
    print("statistically significant — optimized Transformer training reduces but")
    print("does not eliminate English-centric bias.")

    # 7. Summary statistics
    print("\n\n7. SUMMARY STATISTICS")
    print("-" * 80)

    print("\nError counts by direction:")
    for direction in ['Spanish→Greek', 'Greek→Spanish', 'Combined']:
        total = total_sentences['per_direction'] if direction != 'Combined' else total_sentences['combined']
        print(f"\n{direction} (out of {total}):")
        for system in ['Google', 'DeepL Classic', 'DeepL Next Gen']:
            errors = bias_errors[direction][system]
            rate = errors / total
            print(f"  {system:20s}: {errors:3d} errors ({rate:5.1%})")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
