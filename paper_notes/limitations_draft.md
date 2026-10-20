# Draft text for the Limitations section (to be edited by the authors)

Status: draft wording collected from decisions made during the revision; the paper text itself is not in this repository.

1. **Closed systems are a dated snapshot.** Google Translate and DeepL (Classic and Next Gen) were queried on the dates recorded with the data. Commercial systems change without notice, and the vendor's own documentation announces the replacement of the Classic model by the Next-Gen model; the labels "Classic" and "Next Gen" therefore describe the systems as observed, not as available today. Only the open-weight systems (NLLB-200 distilled 1.3B, EuroLLM-9B-Instruct Q4_K_M, OPUS-MT controls) are reproducible, with pinned revisions.

2. **Targeted, not representative, sentences.** The 112 sentences were designed to be English-lossy (they carry a feature that English drops or merges). Reported rates are failure rates on such structures, not general bias rates.

3. **Pivot candidates are model-generated.** The 16 candidate translations and back-translations were produced by a language model (Claude), identically for all systems; the black-box analysis therefore depends on that model's renderings. Exact-string matching has low sensitivity, so the gender-marker criterion is the primary analysis.

4. **Quantization.** EuroLLM-9B-Instruct was run as a GGUF Q4_K_M re-quantization (greedy decoding, llama.cpp), not at full precision.

5. **Polysemy evidence is indicative only.** An exploratory run of a second NLLB checkpoint gave the same gender pattern but a different outcome on the polysemy items; polysemy is therefore presented as indicative rather than conclusive.

6. **Second language pair (French–Polish).** Source renderings, gold, pivot translations and annotation were produced without native-speaker involvement (gold may be checked by a native speaker); sentences without a gender signal in the source are not scored and are reported separately. The study is labelled an exploratory replication on open systems only.

7. **Single run, one annotator for the primary study.** (Edit if a second annotator and κ become available.)

8. **Same sentences reused in the second pair.** The French and Polish source sentences are the same 56-per-direction set carried over, even where a feature does not transfer (e.g. Spanish *nosotras* vs French *nous*). This was a deliberate choice: to keep the dataset as specific and targeted as possible and not to introduce a second design. Sentences that lose their gender signal are labelled and reported separately.
