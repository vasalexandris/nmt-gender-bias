# French<->Polish annotation status

Provisional. The annotations (`FR_PL_annotations.csv`, `PL_FR_annotations.csv`) were produced blind to system identity and in randomised order. A random sample of 30 rows (24 scored, 6 N) was reviewed by the authors on 2026-10-22 without objections, and the rows on the number/formality of "you" were re-checked for consistency; EuroLLM "W porządku." (FR->PL IDs 49, 50) was set to 0 by the authors.

Decision (authors, 2026-10-22): the annotations are locked **after** the authors have reviewed the gold translations (`gold_review_sheet.xlsx`). If a gold translation changes, the annotations that depended on it are re-examined before locking.

## Update 2026-10-22: gold review of the FR->PL direction
The authors reviewed the Polish gold (table `fr-pl_gold_translations.csv`, with comments by a second reviewer) and agreed with it. Corrections applied: ID2 ("i", single "One"), ID5 ("zdecydowały się"), ID32/33 ("sobie nawzajem" / "się nawzajem"), ID39 ("Tamta ławka."), ID49/50 ("Dobrze się masz/macie."), ID51/52 (pronoun "Wy" dropped), IDs 21-23 (both "Jej" and "Jego" accepted). The French gold of the PL->FR direction has **not** been reviewed.

Annotation consequence: the dataset convention that a gender- and person-neutral possessive ("su", "son") accepts any 3rd/2nd-person rendering was applied consistently: OPUS-MT controls ES->EL IDs 21-23 and FR->PL IDs 21-23 set to 0 (14 French-Polish rows and 8 control rows changed). Possessives with an explicit possessor in the source (PL->FR, "Jej") are unaffected.
