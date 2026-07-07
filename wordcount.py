
import pandas as pd
import re
from dictionary_utils import load_dictionaries, DIMS



# Einlesen
data = pd.read_excel('Prompts.xlsx')
data = data[['VP', 'Aggregiert']]

# Dictionaries laden
dictionaries = load_dictionaries()
dict_care      = dictionaries['Care']
dict_fairness  = dictionaries['Fairness']
dict_loyalty   = dictionaries['Loyalty']
dict_authority = dictionaries['Authority']
dict_purity    = dictionaries['Purity']

#Alle Umlaute in den Dictionaries durch ae, oe, ue ersetzen
def normalize_umlaute(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('ß', 'ss')
    return text

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = normalize_umlaute(text)
    text = re.sub(r'[^\w\s]', '', text)  
    return text.split()

# Generische Scoring-Funktion
def score_dict(text, dictionary):
    if pd.isna(text):
        raise ValueError("Text is null")
    words = preprocess_text(text)
    count = 0
    for pattern in dictionary:
        if pattern.endswith('*'):
            stem = pattern.rstrip('*')
            count += sum(1 for w in words if w.startswith(stem))
        else:
            count += sum(1 for w in words if w == pattern)
    return count

#generische Matching-Funktion
def get_matches(text, dictionary):
    if pd.isna(text):
        return ''
    words = preprocess_text(text)
    matches = []
    for pattern in dictionary:
        if pattern.endswith('*'):
            stem = pattern.rstrip('*')
            matches += [w for w in words if w.startswith(stem)]
        else:
            matches += [w for w in words if w == pattern]
    return ', '.join(matches)

#Scores berechnen
data['score_care']      = data['Aggregiert'].apply(lambda t: score_dict(t, dict_care))
data['score_fairness']  = data['Aggregiert'].apply(lambda t: score_dict(t, dict_fairness))
data['score_loyalty']   = data['Aggregiert'].apply(lambda t: score_dict(t, dict_loyalty))
data['score_authority'] = data['Aggregiert'].apply(lambda t: score_dict(t, dict_authority))
data['score_purity']    = data['Aggregiert'].apply(lambda t: score_dict(t, dict_purity))


#Matches listen
data['matches_care']      = data['Aggregiert'].apply(lambda t: get_matches(t, dict_care))
data['matches_fairness']  = data['Aggregiert'].apply(lambda t: get_matches(t, dict_fairness))
data['matches_loyalty']   = data['Aggregiert'].apply(lambda t: get_matches(t, dict_loyalty))
data['matches_authority'] = data['Aggregiert'].apply(lambda t: get_matches(t, dict_authority))
data['matches_purity']    = data['Aggregiert'].apply(lambda t: get_matches(t, dict_purity))


#Als Excel speichern
data[['VP', 'score_care', 'matches_care',
           'score_fairness', 'matches_fairness',
           'score_loyalty', 'matches_loyalty',
           'score_authority', 'matches_authority',
           'score_purity', 'matches_purity']].to_excel('Scoring_Ergebnisse.xlsx', index=False)