import pandas as pd
from scipy import stats

#Einlesen
data = pd.read_excel('Scoring_Ergebnisse.xlsx')
data = pd.read_excel('MFQ_Ergebnisse.xlsx')

# MFQ Platzhalter (später mit SoSci-Daten ersetzen)
data['mfq_care']      = None
data['mfq_fairness']  = None
data['mfq_loyalty']   = None
data['mfq_authority'] = None
data['mfq_purity']    = None

# Z-Normalisierung der Häufigkeitsvektoren
score_cols = ['score_care', 'score_fairness', 'score_loyalty', 'score_authority', 'score_purity']
for col in score_cols:
    data[col + '_z'] = stats.zscore(data[col])

# Korrelation pro VP (Platzhalter - erst wenn MFQ-Daten vorhanden)

mfq_cols = ['mfq_care', 'mfq_fairness', 'mfq_loyalty', 'mfq_authority', 'mfq_purity']
score_z_cols = ['score_care_z', 'score_fairness_z', 'score_loyalty_z', 'score_authority_z', 'score_purity_z']

def correlate_vectors(row):
    mfq_vec   = row[mfq_cols].values.astype(float)
    score_vec = row[score_z_cols].values.astype(float)
    r, p = stats.pearsonr(mfq_vec, score_vec)
    return r

data['pearson_r'] = data.apply(correlate_vectors, axis=1)

#Korrelation mitteln
mean_r = data['pearson_r'].mean()
print(f'Mean Pearson r across participants: {mean_r}')

#Ergebnisse speichern
data.to_excel('Wordcounting_Analysis.xlsx', index=False)