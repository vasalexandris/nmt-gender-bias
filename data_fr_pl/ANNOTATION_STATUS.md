# French<->Polish annotation status

**Final** (locked 2026-10-22).

Process: annotations (`FR_PL_annotations.csv`, `PL_FR_annotations.csv`) were produced blind to system identity and in randomised order, using the rule *1 = gender lost or changed, or tense/number/person changed, or polysemous sense lost; purely lexical differences = 0; a gender- and person-neutral possessive accepts any 3rd/2nd-person rendering*. A random sample of 30 rows (24 scored, 6 N) was reviewed by the authors without objections; the rows on the number/formality of "you" were re-checked for consistency; EuroLLM "W porządku." (FR->PL IDs 49, 50) was set to 0 by the authors.

Gold review (no native speaker was available):
- FR->PL (Polish gold): reviewed by the authors using a table with comments from a second reviewer; corrections applied to ID2, 5, 21-23, 32, 33, 39, 49-52 (gold only; system outputs and the annotations of gender were unaffected).
- PL->FR (French gold): reviewed by the authors using an analysis by another AI; no change needed. Notes added for IDs 18-21 (French cannot mark the possessor's gender).
- Observed in the review and kept: some Polish source sentences (the strict renderings of the original Greek sentences) are slightly unnatural ("nauczała lekcji", "Muchy czasu", "bez dzisiaj"); they are used as-is by design. "Muchy roku / Muchy czasu" are the original polysemy item on "flies".

Consequence for the paper: gold and annotations of the French-Polish study were produced without a native speaker; the study is presented as an exploratory replication (see protocol/ and paper_notes/limitations_draft.md).
