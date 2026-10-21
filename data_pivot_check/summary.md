# English-route robustness check with an open model (EuroLLM-9B-Instruct Q4_K_M, greedy)

Question (Reviewer J: the pivot generator, Claude, may have English-centric tendencies of its own): does an independent open model, asked to translate source -> English -> target, reproduce the information loss found in the Claude-generated English route?

Data: `ES_EL_english_route_eurollm.csv`, `EL_ES_english_route_eurollm.csv` (112 sentences); script `code/run_pivot_check_english.py`; the prompt is archived in the script.

## 1. String-level agreement (strict)
- English intermediate text identical to Claude's: 60/112 (ES->EL 34/56, EL->ES 26/56).
- Back-translation identical to Claude's (case/punctuation-insensitive): 36/112 = 32% (18/56 in each direction).
Exact-string agreement is low because of lexical variation (e.g. "ποινή" vs "ετυμηγορία", "παρακολούθησαν το συνέδριο" vs "τη διάσκεψη"). Exact-match scores therefore depend on the generator.

## 2. Feature-level agreement on the diagnostic sentences (judged by the annotator)
| Feature | Reproduced by EuroLLM's English route |
|---|---|
| Gender loss (masculine default / feminine marker lost), 12 sentences (ES->EL 2, 6, 10, 15, 17, 19, 28, 32, 33, 34, 35; EL->ES 22) | 12/12 |
| Polysemy: ES->EL 38 (bats -> animal), 40 (seal -> stamp) | 2/2 |
| Number/formality of "you" collapse (ES->EL 19, 49, 53; EL->ES 15, 50, 51, 52, 56) | 7/8 (ES->EL 49 differs: EuroLLM keeps the singular) |

## 3. Reading
The information loss is a property of the English intermediate step, not of one generator: an independent open model reproduces the same losses on 21 of 22 feature checks, while the exact wording differs in most sentences. Consequently the feature-level analysis (gender marker, number/formality, sense) is robust to the choice of generator, whereas the exact-string pivot score is not and should be reported with that caveat.

Limits: EuroLLM is itself an English-capable LLM, so this check shows that English-mediated translation produces these losses, not that it is independent of English in every sense; judgments are by the annotator; one run.
