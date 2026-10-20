"""Build data_fr_pl/{FR_PL,PL_FR}_pivots.csv: 16 candidate-language translations of each source sentence and their back-translations
into the target language (written by Claude; see bt_fr_pl.py / bt_pl_fr.py for the conventions). Run from the repository root."""
import csv, sys
sys.path.insert(0, 'code/fr_pl')
from bt_fr_pl import LANGS, BT as BT_FR_PL
from bt_pl_fr import BT as BT_PL_FR
EXIST = [l for l in LANGS if l not in ('Spanish', 'Greek')]

# New candidates (Spanish, Greek) = strict translations of the French / Polish source; everything else reused from the existing pivot files.
ES_N_FR = {1:"Hemos terminado nuestros estudios.",9:"Estos científicos han descubierto una nueva vacuna.",10:"Estos periodistas informaron desde la zona de guerra.",
 11:"Estos científicos lideran el proyecto.",12:"Estos médicos tienen mucha experiencia.",13:"Estos arquitectos diseñaron varios edificios.",
 14:"Los arquitectos presentaron el proyecto innovador.",15:"Un juez dictó la sentencia ayer.",17:"Todos los ministros asistieron a la conferencia.",
 19:"¿Estás cansado después del trabajo?",20:"¿Estáis cansados del viaje?",36:"Yo mismo repararé el coche.",37:"Nosotros mismos organizaremos el evento."}
EL_N_FR = {1:"Έχουμε ολοκληρώσει τις σπουδές μας.",9:"Αυτοί οι επιστήμονες ανακάλυψαν ένα νέο εμβόλιο.",10:"Αυτοί οι δημοσιογράφοι ανέφεραν από τη ζώνη του πολέμου.",
 11:"Αυτοί οι επιστήμονες ηγούνται του έργου.",12:"Αυτοί οι γιατροί έχουν μεγάλη εμπειρία.",13:"Αυτοί οι αρχιτέκτονες σχεδίασαν πολλά κτίρια.",
 14:"Οι αρχιτέκτονες παρουσίασαν το καινοτόμο έργο.",15:"Ένας δικαστής εξέδωσε την απόφαση χθες.",17:"Όλοι οι υπουργοί παρευρέθηκαν στη διάσκεψη.",
 19:"Είσαι κουρασμένος μετά τη δουλειά;",20:"Είστε κουρασμένοι από το ταξίδι;",36:"Εγώ ο ίδιος θα επισκευάσω το αυτοκίνητο.",37:"Εμείς οι ίδιοι θα οργανώσουμε την εκδήλωση.",
 40:"Η αρχαία φώκια",51:"Εσείς είστε πολύ καλές.",52:"Εσείς είστε πολύ καλοί.",}
ES_N_PL = {10:"Los abogados defendieron el caso.",11:"El arquitecto diseñó el nuevo edificio.",34:"Se apoyan mutuamente en el trabajo.",42:"Alex vino sin hoy.",
 43:"Jordan trabaja como ingeniero.",44:"Morgan quiere comprar un coche nuevo.",51:"Estáis bien.",52:"Jugáis muy bien."}
EL_N_PL = {11:"Ο αρχιτέκτονας σχεδίασε το νέο κτίριο.",33:"Η μία βοηθά πάντα την άλλη.",34:"Υποστηρίζουν ο ένας τον άλλον στη δουλειά.",
 42:"Ο Alex ήρθε χωρίς σήμερα.",43:"Ο Jordan εργάζεται ως μηχανικός.",44:"Ο Morgan θέλει να αγοράσει νέο αυτοκίνητο."}

def rd(f): return list(csv.DictReader(open(f, encoding='utf-8-sig')))
def build(direction, nextgen, srccol, goldcol_es_or_el, BT, es_over, el_over, target):
    base = rd(nextgen); sig = {int(r['Original_ID']): r for r in rd('data_fr_pl/source_sentences_as_is.csv') if r['Direction'] == direction}
    rows = []
    for x in base:
        i = int(x['ID']); s = sig[i]
        # Spanish / Greek candidate texts
        if direction == 'FR_PL':
            es = es_over.get(i, x['Source_Spanish']); el = el_over.get(i, x['Gold_Greek'])
        else:
            es = es_over.get(i, x['Gold_Spanish']); el = el_over.get(i, x['Source_Greek'])
        cand = {l: x[f'Pivot_{l}_Translation'] for l in EXIST}; cand['Spanish'] = es; cand['Greek'] = el
        variants, assign, *alt = BT[i]; alt = alt[0] if alt else {}
        row = {'ID': i, 'Direction': direction, 'Source_As_Is': s['Source_As_Is'], 'Gold': s['Gold'], 'Signal_In_Source': s['Signal_In_Source']}
        for k, l in enumerate(LANGS):
            row[f'Pivot_{l}_Translation'] = cand[l]
            row[f'Pivot_{l}_BackTrans_{target}'] = variants[assign[k]]
            row[f'Pivot_{l}_BackTrans_Alt'] = ' | '.join(variants[c] for c in alt.get(k, ''))
        rows.append(row)
    return rows
fr = build('FR_PL', 'ES_EL_deepl_nextgen.csv', 'Source_Spanish', None, BT_FR_PL, ES_N_FR, EL_N_FR, 'Polish')
pl = build('PL_FR', 'EL_ES_deepl_nextgen.csv', 'Source_Greek', None, BT_PL_FR, ES_N_PL, EL_N_PL, 'French')
for name, rows in (('FR_PL', fr), ('PL_FR', pl)):
    w = csv.DictWriter(open(f'data_fr_pl/{name}_pivots.csv', 'w', encoding='utf-8-sig', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(name, len(rows), 'rows')
