#!/usr/bin/env python3
"""
Statistical Analysis for Multilingual NMT Gender Bias Study

This script computes:
1. Softmax probabilities from matching scores
2. Cohen's h effect sizes
3. Summary statistics
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import json

# ============================================================================
# Data: Pivot Language Matching Scores
# ============================================================================

# Spanish → Greek direction (out of 56 sentences)
spanish_greek_scores = {
    'Google Translate': {
        'English': 29, 'German': 13, 'Swedish': 20, 'Turkish': 19,
        'Chinese': 18, 'Swahili': 18, 'Hungarian': 17, 'Italian': 13,
        'French': 14, 'Polish': 13, 'Arabic': 13, 'Hebrew': 13, 'Hindi': 13
    },
    'DeepL Classic': {
        'English': 22, 'German': 13, 'Swedish': 18, 'Turkish': 18,
        'Chinese': 18, 'Swahili': 18, 'Hungarian': 17, 'French': 16,
        'Polish': 13, 'Italian': 12, 'Arabic': 12, 'Hebrew': 12, 'Hindi': 12
    },
    'DeepL Next Gen': {
        'Polish': 24, 'Italian': 23, 'Arabic': 24, 'Hebrew': 24,
        'Hindi': 24, 'German': 11, 'French': 18, 'Swedish': 11,
        'English': 10, 'Turkish': 8, 'Hungarian': 8, 'Chinese': 8, 'Swahili': 8
    }
}

# Greek → Spanish direction (out of 56 sentences)
greek_spanish_scores = {
    'Google Translate': {
        'English': 28, 'Turkish': 24, 'Swedish': 23, 'Russian': 21,
        'Hungarian': 19, 'Swahili': 19, 'Japanese': 18, 'Chinese': 18,
        'Albanian': 16, 'Arabic': 15, 'Hindi': 14, 'Hebrew': 14,
        'German': 13, 'French': 12, 'Polish': 12
    },
    'DeepL Classic': {
        'English': 20, 'Swedish': 17, 'Turkish': 17, 'Chinese': 17,
        'Japanese': 16, 'Swahili': 15, 'Hungarian': 15, 'Russian': 15,
        'Albanian': 14, 'Hindi': 13, 'German': 12, 'Polish': 12,
        'Arabic': 12, 'French': 11, 'Italian': 11
    },
    'DeepL Next Gen': {
        'Hebrew': 25, 'Italian': 24, 'Arabic': 24, 'French': 23,
        'Polish': 22, 'Albanian': 21, 'German': 20, 'Hindi': 19,
        'Chinese': 18, 'Russian': 14, 'Japanese': 14, 'Turkish': 13,
        'Hungarian': 13, 'Swedish': 12, 'Swahili': 12
    }
}

# Gender bias error counts
bias_errors = {
    'Spanish→Greek': {'Google': 45, 'DeepL Classic': 42, 'DeepL Next Gen': 5},
    'Greek→Spanish': {'Google': 40, 'DeepL Classic': 37, 'DeepL Next Gen': 3},
    'Combined': {'Google': 85, 'DeepL Classic': 79, 'DeepL Next Gen': 8}
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
            
            # Sort by probability (descending)
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n  {system}:")
            for lang, prob in sorted_probs[:5]:  # Top 5
                print(f"    {lang:12s}: {prob:6.1%} ({scores[lang]}/56 matches)")
            
            # English probability
            if 'English' in probs:
                print(f"    {'English':12s}: {probs['English']:6.1%} *** ENGLISH ***")
    
    # 2. Combined direction analysis (average of both directions)
    print("\n\n2. COMBINED DIRECTION SOFTMAX (REPORTED IN PAPER)")
    print("-" * 80)
    
    # Average English scores across both directions
    for system in ['Google Translate', 'DeepL Classic', 'DeepL Next Gen']:
        # Get English scores from both directions
        es_el_english = spanish_greek_scores[system].get('English', 0)
        el_es_english = greek_spanish_scores[system].get('English', 0)
        
        # Combine all scores from both directions
        all_scores = {}
        
        # Add Spanish→Greek scores
        for lang, score in spanish_greek_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score
        
        # Add Greek→Spanish scores
        for lang, score in greek_spanish_scores[system].items():
            all_scores[lang] = all_scores.get(lang, 0) + score
        
        # Compute softmax on combined scores
        combined_probs = softmax(all_scores)
        
        print(f"\n{system}:")
        sorted_probs = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        
        for lang, prob in sorted_probs[:5]:
            print(f"  {lang:12s}: {prob:6.1%}")
        
        if 'English' in combined_probs:
            print(f"  {'English':12s}: {combined_probs['English']:6.1%} *** REPORTED VALUE ***")
    
    # 3. Cohen's h effect sizes
    print("\n\n3. COHEN'S h EFFECT SIZES")
    print("-" * 80)
    
    # Bias rates (combined directions)
    google_rate = bias_errors['Combined']['Google'] / total_sentences['combined']
    deepl_c_rate = bias_errors['Combined']['DeepL Classic'] / total_sentences['combined']
    deepl_ng_rate = bias_errors['Combined']['DeepL Next Gen'] / total_sentences['combined']
    
    print(f"\nBias rates (combined):")
    print(f"  Google Translate: {google_rate:.1%} ({bias_errors['Combined']['Google']}/112)")
    print(f"  DeepL Classic:    {deepl_c_rate:.1%} ({bias_errors['Combined']['DeepL Classic']}/112)")
    print(f"  DeepL Next Gen:   {deepl_ng_rate:.1%} ({bias_errors['Combined']['DeepL Next Gen']}/112)")
    
    # Effect sizes
    h_google_ng = cohens_h(google_rate, deepl_ng_rate)
    h_deepl_c_ng = cohens_h(deepl_c_rate, deepl_ng_rate)
    h_google_deepl_c = cohens_h(google_rate, deepl_c_rate)
    
    print(f"\nEffect sizes (Cohen's h):")
    print(f"  Google → DeepL Next Gen:        h = {h_google_ng:5.2f} (huge)")
    print(f"  DeepL Classic → DeepL Next Gen: h = {h_deepl_c_ng:5.2f} (huge)")
    print(f"  Google ↔ DeepL Classic:         h = {h_google_deepl_c:5.2f} (negligible)")
    
    # Interpretation
    print("\nInterpretation (Cohen, 1988):")
    print("  |h| < 0.2:        small")
    print("  0.2 ≤ |h| < 0.5:  small to medium")
    print("  0.5 ≤ |h| < 0.8:  medium to large")
    print("  |h| ≥ 0.8:        large/huge")
    
    # 4. Bias reduction
    print("\n\n4. BIAS REDUCTION")
    print("-" * 80)
    
    reduction_google = google_rate / deepl_ng_rate
    reduction_deepl_c = deepl_c_rate / deepl_ng_rate
    
    print(f"\nGoogle → DeepL Next Gen:        {reduction_google:.1f}× reduction")
    print(f"DeepL Classic → DeepL Next Gen: {reduction_deepl_c:.1f}× reduction")
    
    # 5. Summary statistics
    print("\n\n5. SUMMARY STATISTICS")
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
