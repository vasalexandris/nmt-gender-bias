# Pre-specified protocol: replication of the English-pivot analysis on French ↔ Polish (open, reproducible systems only)

Status: **DRAFT, not yet frozen.** It becomes binding when committed with the line `FROZEN: <date>` below, *before* any
FR/PL sentence is translated by any system. After freezing, every deviation is logged in `protocol/DEVIATIONS.md`.

FROZEN: (not frozen)

## 1. Purpose and scope

The primary study (Spanish ↔ Greek, 112 sentences, five systems) used closed commercial systems. Those systems are
snapshots at a given date; they cannot be re-run identically. This protocol tests whether the *method* and the
*regime-not-architecture* finding generalise to a second language pair, using **only reproducible systems**
(pinned open weights). It does **not** attempt to replicate the closed-system results (Google Translate, DeepL).

Claims this study can support: (a) the 16-candidate pivot method and the gender-marker criterion work in a second pair;
(b) an English-pivoted pipeline shows markedly more gender loss than a direct bilingual one; (c) NLLB-200 and EuroLLM
behave relative to these controls as in Spanish–Greek.
Claims it cannot support: that Google/DeepL route through English in French–Polish; population-level bias rates.

## 2. Design principle of the sentence set

The 112 sentences are *English-lossy by design*: each carries a feature (grammatical gender of a 3rd-person plural
pronoun, article, adjective or participle; number/formality of "you"; a lexical item whose English translation is
ambiguous) that English drops or merges. The set is therefore targeted, not representative; reported rates are
"failure rates on English-lossy structures", not general bias rates.

## 3. Sentence adaptation (FR and PL)

- The existing sentences are reused, not redesigned. The 56 Spanish sources (ES→EL) are translated into French (FR→PL) and the 56 Greek
  sources (EL→ES) into Polish (PL→FR) by a native speaker of the source language, keeping `Category` and `Gender_Bias_Focus` labels.
  Only items that cannot be carried over (see below) are replaced by a new item of the same category. The target-language gold is the only
  text written from scratch.
- Adaptation criteria, in order: (1) the source sentence must carry the target feature in a form English would drop
  or merge (e.g. French *elles*, *celles*, participle agreement; Polish past-tense gender, *one/oni*); (2) same grammatical
  category as the original; (3) same length and register; (4) polysemy items use a source word with two distinct target
  translations that coincide in English (new lexical items, listed with both senses).
- Items with no 1-to-1 equivalent (e.g. features of Spanish *nosotras*, Greek neuter) are **replaced** by an item of the same
  category or **dropped**, and logged in `data_fr_pl/adaptation_log.csv` (columns: original ID, direction, status
  {kept, adapted, replaced, dropped}, reason). The final N per direction is reported; a minimum of 45 per direction
  is required for the study to proceed.
- A gold translation is written by a native speaker of the target language. Gold is fixed before any system output is seen.
- The English route of every adapted sentence is checked (Claude-generated, as in the primary study): sentences
  where the English route *retains* the tested feature are flagged and excluded from the diagnostic subset.
- The sentence set and gold are committed and tagged **before** any system is run.

## 4. Systems (all open, all pinned)

| System | Setting | Revision / pin |
|---|---|---|
| NLLB-200 distilled 1.3B | beam 5, max_length 200, as `code/run_nllb.py` | `7be3e24664b38ce1cac29b8aeed6911aa0cf0576` |
| EuroLLM-9B-Instruct (GGUF Q4_K_M), llama.cpp | greedy (temperature 0), stateless per-sentence prompt, as `code/run_eurollm.py` | GGUF repo commit `55e5590cc457e42ed197c34c5b10403472b05d54` |
| OPUS-MT direct | beam 5, max_length 200 | `fr-pl` c82045f9911a87b048bed59c91379032f09ab486; `pl-fr` d6cf8c144f2c99600eb8e794f7f9d93b24fac22b |
| OPUS-MT cascade via English | each hop beam 5; `>>pol<<` prefix on the en→pl hop | `fr-en` c4aed37b318c763fd177aa449b44e3b783cc6c02; `en-zlw` 92ffd5aa936924688bd715b26d6bb75393082c03; `pl-en` 7f2bb874fdfb6139f9842b91a9b75c4a6c93401c; `en-fr` dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 |
| OPUS-MT cascade via Spanish (gender-preserving control) | each hop beam 5 | `fr-es` f4263aff464dbfb2650a4b1793baf6f82b91ae91; `es-pl` 92eef1033791189c896b1e0d9fbaf1227528f971; `pl-es` b97a98faa36c86f27e58f7b7f528175fae32a20d; `es-fr` f8be65b3aff01c69f3b9092a4ff81b873612140d |

Known limitations to be stated in the paper: `en-zlw` is a multilingual West-Slavic model, not bilingual; OPUS-MT models are small
and of uneven quality; single run per system; `fr-en` was updated on the Hub in Nov 2025 (pinned revision above is the one used).

## 5. Pivot analysis (black box)

- 16 candidate pivot languages, identical for all systems. Relative to the primary study, **French and Polish are removed**
  (they are the pair) and **Spanish and Greek are added**: English, Swedish, Turkish, German, Russian, Hungarian, Italian,
  Albanian, Chinese, Japanese, Arabic, Hebrew, Hindi, Swahili, Spanish, Greek. Spanish acts as a gender-preserving candidate.
- Pivot translations (source→L) and back-translations (L→target) are generated by the same Claude model and prompt template as in the primary
  study (model ID and date recorded at generation); this dependence is a stated limitation. A robustness check regenerates the
  English route with one independent system (an open model) and reports agreement.
- Pivot match = exact string match after case-insensitive comparison, as in the primary study; **plus** the
  gender-marker criterion (Section 6), which is the primary analysis because exact matching has low sensitivity.

## 6. Annotation and primary measures

- **Gender_Bias** (0/1), same rule as the primary study: 1 = gender lost or changed, or tense/number/person changed,
  or the polysemous sense lost; purely lexical differences = 0. A short justification is recorded for every 1.
- Two annotators who are native speakers of the target language. Annotation is done blind to system identity,
  in randomised order. The second annotator labels a random 30% (stratified by category); Cohen's κ is reported
  before reconciliation; disagreements are resolved by discussion and logged.
- **Diagnostic subset** (defined before outputs are seen, on the English route alone): sentences where (a) the English route is
  unique among the 16 candidates and (b) it diverges from gold on the tested feature. A system "matches" when its output carries the English
  route's biased marker (not necessarily the same string).
- Primary measures: (M1) bias rate per system and direction with 95% Wilson intervals; (M2) diagnostic-subset match rate per system
  with Wilson intervals; (M3) the English-route share among biased vs non-biased sentences.

## 7. Pre-specified hypotheses and confirmation criteria

Thresholds are set from the Spanish–Greek values (English cascade 83.9%/71.4% vs direct 44.6%; diagnostic 15/15 vs 5/13), so they are
replication targets, not discovery claims.

- **H1 (positive control).** The English cascade's diagnostic match rate is ≥ 80% in each direction. Not met → the detector is
  not validated for this pair and pivot conclusions are withheld.
- **H2 (regime, not architecture).** In FR→PL and PL→FR the English cascade's bias rate exceeds the direct model's by ≥ 20
  percentage points, and the Spanish cascade's bias rate is lower than the English cascade's. Reported per direction; confirmed only if both hold.
- **H3 (open-weight models).** NLLB and EuroLLM diagnostic match rates are reported against the English cascade, the Spanish
  cascade and direct. No threshold: descriptive, given n. A statement of the form "NLLB is closer to X" requires non-overlapping
  intervals or is phrased as inconclusive.
- A paired exact test (McNemar) is reported for system pairs on identical sentences; no correction claims are made beyond the three hypotheses above.

## 8. What is reported regardless of outcome

All results, including failures of H1–H3, are reported. The adaptation log, gold, outputs, annotations, κ, and scripts are released
with the repo. Exploratory analyses are labelled as such and kept separate from Sections 6–7.

## 9. Outstanding decisions before freezing

1. Name and qualifications of the native annotators (FR, PL) and of the sentence adapters.
2. Whether to include NLLB-200 3.3B as a second checkpoint (it shares training data with 1.3B; not an independent replicate).
3. Independent system for the English-route robustness check (Section 5).
4. Final N per direction after adaptation.
