import pandas as pd
from scipy import stats

#Einlesen
data_countscore = pd.read_excel('Scoring_Ergebnisse.xlsx')
data_MFQ = pd.read_excel('MFQ_Ergebnisse.xlsx') #existiert noch nicht, Platzhalter
data_embscore = pd.read_excel('Embedding_Scores.xlsx')

#Zusammenführungen
data_counting= pd.merge(data_countscore, data_MFQ, on='VP', how='inner')
data_embedding = pd.merge(data_embscore, data_MFQ, on='VP', how='inner')

# Z-Normalisierung der Häufigkeitsvektoren
rawscore_cols = ['score_care', 'score_fairness', 'score_loyalty', 'score_authority', 'score_purity']
for col in rawscore_cols:
    data_counting[col + '_z'] = stats.zscore(data_counting[col])

# Spaltennamen für die Korrelation
mfq_cols = ['mfq_care', 'mfq_fairness', 'mfq_loyalty', 'mfq_authority', 'mfq_purity']
score_z_cols = ['score_care_z', 'score_fairness_z', 'score_loyalty_z', 'score_authority_z', 'score_purity_z']
emb_cols = ['emb_care', 'emb_fairness', 'emb_loyalty', 'emb_authority', 'emb_purity']

# generische Funktion zur Berechnung der Korrelation zwischen zwei Vektoren
def correlate_vectors(row, score_cols):
    mfq_vec   = row[mfq_cols].values.astype(float)
    score_vec = row[score_cols].values.astype(float)
    r, p = stats.pearsonr(mfq_vec, score_vec)
    return r

# Aufruf für counting und embedding
data_counting['pearson_r'] = data_counting.apply(lambda row: correlate_vectors(row, score_z_cols), axis=1)
data_embedding['pearson_r'] = data_embedding.apply(lambda row: correlate_vectors(row, emb_cols), axis=1)

#Korrelationen mitteln
mean_r_counting = data_counting['pearson_r'].mean()
mean_r_embedding = data_embedding['pearson_r'].mean()
print(f'Mean Pearson r for counting: {mean_r_counting}')
print(f'Mean Pearson r for embedding: {mean_r_embedding}')

#Ergebnisse speichern
data_counting.to_excel('Wordcounting_Analysis.xlsx', index=False)
data_embedding.to_excel('Embedding_Analysis.xlsx', index=False)
