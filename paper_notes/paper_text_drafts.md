# Draft paper text (for the authors to edit)

Status: drafts written from the TACL decision letter and reviews. The manuscript itself is not in this repository, so wording must be merged by hand. Bibliographic details were checked against the ACL Anthology / arXiv pages; two points need the authors' confirmation (marked **CHECK**).

## 1. Title options (state the language pair; do not claim internal representations)

The reviews objected that the evidence is behavioural, not representational. The title should therefore say what was measured.

1. *Is English the Hidden Pivot? A Black-Box Study of Gender and Number Loss in Greek–Spanish Machine Translation*
2. *English-Mediated Information Loss in Greek↔Spanish Machine Translation: Evidence from Commercial and Open Systems*
3. *The English Trap in Greek–Spanish Translation: Black-Box Evidence of English-Mediated Gender Loss* (closest to the original title)

## 2. Related work (draft)

**Latent language and shared representations.** A line of work analyses the internal representations of multilingual models. For multilingual NMT, Kudugunta et al. (2019) used SVCCA on a 103-language model and found that encoder representations cluster by linguistic similarity and depend on the target language. For multilingual language models, Wendler et al. (2024) applied the logit lens to Llama-2 and found that the intermediate layers pass through a concept space that leans toward English before the output language is selected; Schut et al. (2025) report that key decisions are made in a representation space closer to English, using the logit lens and steering vectors; and Zhong et al. (2025) show that English-centric models use English as their latent language, whereas non-English-centric models use several latent languages and favour the one closest to the input and output languages. These studies probe open-weight models from the inside.

**Our contribution relative to this work.** We do not probe representations. We measure behaviour: sentences designed so that English drops or merges a distinction (grammatical gender of third-person plural forms, number and formality of "you", lexically ambiguous words), and a black-box procedure that compares each system's output with the output of 16 candidate pivot routes. The evidence is therefore behavioural and complements, but does not replace, the representational evidence above. Our results are consistent with English-mediated information loss in some systems and absent in others; they do not show that a shared latent space is aligned with English.

**Gender bias and architecture in machine translation.** Costa-jussà et al. (2022, AAAI) studied how multilingual architecture affects gender bias and found that language-specific encoder–decoders show less bias than a shared encoder–decoder, which is closely related to our comparison of translation regimes (bilingual versus English-pivoted systems; see the controls with OPUS-MT).

**LLM-based translation.** Prompting large language models for translation was studied by Vilar et al. (2023) for PaLM. Our DeepL Next Gen and EuroLLM results are positioned as observations about particular systems, not as a claim about LLMs in general.

References to add:
- Kudugunta, S., Bapna, A., Caswell, I., Arivazhagan, N., Firat, O. (2019). Investigating Multilingual NMT Representations at Scale. EMNLP-IJCNLP 2019, pp. 1565–1575. https://aclanthology.org/D19-1167/
- Wendler, C., Veselovsky, V., Monea, G., West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. ACL 2024 (long), pp. 15366–15394. https://aclanthology.org/2024.acl-long.820/
- Schut, L., Gal, Y., Farquhar, S. (2025). Do Multilingual LLMs Think In English? arXiv:2502.15603 (ICLR 2025 workshop poster). **CHECK** whether a refereed version exists before submission (reviewers ask to cite it instead of the preprint).
- Zhong, C., Liu, Q., Cheng, F., Jiang, J., Wan, Z., Chu, C., Murawaki, Y., Kurohashi, S. (2025). What Language Do Non-English-Centric Large Language Models Think in? Findings of ACL 2025, pp. 26333–26346. https://aclanthology.org/2025.findings-acl.1350/
- Costa-jussà, M. R., Escolano, C., Basta, C., Ferrando, J., Batlle, R., Kharitonova, K. (2022). Interpreting Gender Bias in Neural Machine Translation: Multilingual Architecture Matters. AAAI 2022, pp. 11855–11863.
- Vilar, D., Freitag, M., Cherry, C., Luo, J., Ratnakar, V., Foster, G. (2023). Prompting PaLM for Translation: Assessing Strategies and Performance. ACL 2023 (long), pp. 15406–15427.
- NLLB Team, Costa-jussà, M. R., et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.

## 3. The Costa-jussà et al. (2022) correction

Reviewer C: "The citations (Recent LLM-based translation (Vilar et al., 2023; Costa-jussà et al., 2022) suggests …) are inaccurate. Costa-jussà et al. (2022) uses a standard encoder–decoder NMT model rather than an LLM."

Both Costa-jussà et al. (2022) works are encoder–decoder NMT: NLLB (a sparsely gated mixture-of-experts encoder–decoder; the released dense and distilled checkpoints are also encoder–decoder) and the AAAI 2022 paper on multilingual architecture and gender bias. **CHECK** which one the manuscript cites at that sentence. Suggested fix:
- Keep Vilar et al. (2023) as the example of LLM-based translation, and remove Costa-jussà et al. (2022) from that sentence.
- Cite Costa-jussà et al. (2022, AAAI) where gender bias and architecture are discussed (Section 2 above), and the NLLB Team (2022) where the open NLLB-200 checkpoint is introduced as a standard encoder–decoder NMT system.
- For the LLM-based side of the comparison, say explicitly which systems are LLM-based: EuroLLM-9B (decoder-only) and, as reported by the vendor, DeepL Next Gen.

## 4. Data and code availability statement (draft)

"All data and code are available at https://github.com/vasalexandris/nmt-gender-bias (archived on Zenodo, DOI 10.5281/zenodo.XXXX **CHECK**; MIT licence). The repository contains the 112 Greek–Spanish sentences with gold translations; the outputs of Google Translate and DeepL (Classic and Next Gen), which are dated snapshots of commercial systems and cannot be re-run identically; the outputs, annotations and justifications for the open systems (NLLB-200 distilled 1.3B, EuroLLM-9B-Instruct Q4_K_M); the OPUS-MT control systems; the 16-language pivot translations and back-translations; and the scripts that produce every table. The open systems are specified by pinned model revisions (NLLB-200 distilled 1.3B: facebook/nllb-200-distilled-1.3B@7be3e24664b38ce1cac29b8aeed6911aa0cf0576; EuroLLM-9B-Instruct GGUF: bartowski/EuroLLM-9B-Instruct-GGUF@55e5590cc457e42ed197c34c5b10403472b05d54, Q4_K_M quantisation, greedy decoding) and can be re-run with the scripts in code/. The pivot translations were generated with a commercial language model (Claude), identically for all systems, and are released as data. The second-language-pair replication (French–Polish) follows a protocol frozen before the systems were run (protocol/)."

Things the statement must not omit: the date of the closed-system queries; that the pivot generator's exact prompts and model version/date must be released (currently missing from the repository); and that the French–Polish gold and annotation have not been checked by a native speaker.

## 5. Repository inconsistencies to fix before submission
- `CITATION.cff` is not valid YAML as written (a tab-indented `orcid` line) and uses placeholder ORCID values (`0000-0000-0000-0000`).
- The Zenodo DOI in `CITATION.cff` (10.5281/zenodo.18686006) differs from the one in the README (10.5281/zenodo.18881815).
- The title in `CITATION.cff` and `preferred-citation` is the old title.
