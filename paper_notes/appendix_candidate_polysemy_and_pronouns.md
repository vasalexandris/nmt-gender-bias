# Appendix draft: candidate English-ambiguous items for future probes (NOT part of the 112-sentence dataset)

Status: proposals only; the Spanish/Greek senses below come from the drafting model's knowledge and have **not** been verified against a dictionary or a native speaker. None of these items was run on any system.

## Why these items
A pivot through English loses information that Spanish and Greek keep. Two kinds of loss separate English from other gender-neutral candidate languages (Swedish, Turkish, ...), which also lose grammatical gender and therefore cannot be told apart from English on gender alone:
1. **Lexical ambiguity**: one English word covering two senses that Spanish and Greek express with two words.
2. **Number / formality of "you" and related pronouns**: "you / your / yours / yourself" do not mark number, and "myself / ourselves / themselves" do not mark gender.

## English-ambiguous words (ES / EL both distinguish the senses)
| English | Spanish | Greek |
|---|---|---|
| bat (used in the dataset) | murciélago / bate | νυχτερίδα / ρόπαλο |
| seal (used in the dataset) | foca / sello | φώκια / σφραγίδα |
| nail | uña / clavo | νύχι / καρφί |
| bark | ladrido / corteza | γάβγισμα / φλοιός |
| pupil | alumno / pupila | μαθητής / κόρη (του ματιού) |
| match | cerilla / partido | σπίρτο / αγώνας |
| spring | primavera / muelle / manantial | άνοιξη / ελατήριο / πηγή |
| bow | arco / lazo | τόξο / φιόγκος |
| tie | corbata / empate | γραβάτα / ισοπαλία |
| jam | mermelada / atasco | μαρμελάδα / μποτιλιάρισμα |
| glasses | gafas / vasos | γυαλιά / ποτήρια |
| file | archivo / lima | αρχείο / λίμα |
| date | fecha / cita / dátil | ημερομηνία / ραντεβού / χουρμάς |
| mine | mina / mío | ορυχείο / δικός μου |

Not usable in Greek because one Greek word covers both senses: *crane* (γερανός), *letter* (γράμμα). Usable for Spanish only.

## English pronoun forms that collapse distinctions
- *yours*, *your*, *yourself*, *yourselves* (singular/plural, formal/informal) versus Spanish *tuyo/suyo/vuestro*, *tú mismo/vosotras mismas* and Greek *δικό σου/σας*.
- *myself*, *ourselves*, *themselves*, *herself* (gender of the intensifier) versus Spanish *yo misma/nosotras mismas/ellas mismas* and Greek *η ίδια/οι ίδιες*.
- Items of the existing set that already test this: 26, 27, 30, 32, 36, 37 (intensifiers) and 21–23, 53, 54 (possessives) in the ES→EL set; the corresponding items of the EL→ES set.

## Possible use
Short probe set for a follow-up study, or a paragraph in Future Work. Before any use: verify each sense pair, write minimal sentences without context cues, and add them to the pre-specified protocol before running any system.
