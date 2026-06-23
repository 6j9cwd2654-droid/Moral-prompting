
import pandas as pd
import re


# Einlesen
dict_df = pd.read_excel('mfd_de_referenz.xlsx', sheet_name='MFD_DE')
data = pd.read_excel('Prompts.xlsx')
data = data[['VP', 'Aggregiert']]

# Dictionaries bauen
dict_care      = dict_df[dict_df['Care'] == 1]['match_pattern'].tolist()
dict_fairness  = dict_df[dict_df['Fairness'] == 1]['match_pattern'].tolist()
dict_loyalty   = dict_df[dict_df['Loyalty'] == 1]['match_pattern'].tolist()
dict_authority = dict_df[dict_df['Authority'] == 1]['match_pattern'].tolist()
dict_purity    = dict_df[dict_df['Purity'] == 1]['match_pattern'].tolist()

#Alle Umlaute in den Dictionaries durch ae, oe, ue ersetzen
def normalize_umlaute(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('ß', 'ss')
    return text

# Generische Scoring-Funktion
def score_dict(text, dictionary):
    if pd.isna(text):
        raise ValueError("Text is null")
    text = text.lower()
    text = normalize_umlaute(text)
    words = re.sub(r'[^\w\s]', '', text).split()
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
    text = text.lower()
    text = normalize_umlaute(text)
    words = re.sub(r'[^\w\s]', '', text).split()
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