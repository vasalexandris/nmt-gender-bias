# OPUS-MT control systems

Purpose: test whether the "English signature" in the pivot analysis comes from the *architecture* (transformer) or from the
*training/inference regime* (English involvement). All controls are Marian bilingual transformers (OPUS-MT, Helsinki-NLP),
greedy-free beam search (5), one sentence per call, single run, CPU.

| Control | ES→EL | EL→ES |
|---|---|---|
| Direct bilingual | `opus-mt-es-el` | none usable (no `opus-mt-el-es`) |
| Cascade via English | `opus-mt-es-en` → `opus-mt-en-el` | `opus-mt-ine-en` → `opus-mt-en-es` |
| Cascade via French | `opus-mt-es-fr` → `opus-mt-fr-el` | `opus-mt-el-fr` → `opus-mt-fr-es` |

Pinned revisions: es-en c96e2c53…, en-el 16536739…, es-el a47b45fa…, es-fr f8be65b3…, fr-el b72fefaf…, ine-en 0e2ead27…,
el-fr 8add6b35…, fr-es f4263aff…, en-es 5bc4493d… (full hashes printed by the scripts and in the commit message).

Notes / limitations
- `opus-mt-tc-big-el-en` produced degenerate output (even for its own model-card example) under the transformers versions we
  tried (5.17, 4.46.3); `code/run_opus_controls_part1.py` therefore contains an unusable EL→EN half whose output was discarded.
  EL→EN uses `opus-mt-ine-en` instead (`code/run_opus_controls_part2.py`).
- `Direct_Bias`/`Cascade_*_Bias` are provisional annotations by the same rule as the other systems, pending author review.
- The 15-sentence diagnostic match flags (13 ES→EL, 2 EL→ES) are in `code/opus_control_judgments.py`.
- Exact-string matching against the Claude-generated back-translations has low sensitivity even for a true English cascade
  (it matched 15/56 ES→EL), because the OPUS wording differs from Claude's; use the gender-marker criterion for the controls.
