import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
 
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

#Einlesen
data_countscore = pd.read_excel('Scoring_Ergebnisse.xlsx')
data_MFQ = pd.read_excel('Daten_bereinigt.xlsx', sheet_name='Data')[
    ['participant_id', 'mfq_care_score', 'mfq_fairness_score',
     'mfq_loyalty_score', 'mfq_authority_score', 'mfq_purity_score']
     ]
data_embscore = pd.read_excel('Embedding_Scores.xlsx')

#Zusammenführungen
data_counting= pd.merge(data_countscore, data_MFQ, on='participant_id', how='inner')
data_embedding = pd.merge(data_embscore, data_MFQ, on='participant_id', how='inner')

# Rohe Trefferzahlen relativ zur individuellen Wortzahl statt absolut, da sich sonst
# Textlaenge und tatsaechliche moralische Sprache vermischen (kurze Texte -> wenig Treffer)
rawscore_cols = ['score_care', 'score_fairness', 'score_loyalty', 'score_authority', 'score_purity']
perword_cols = [col + '_perword' for col in rawscore_cols]
for raw_col, perword_col in zip(rawscore_cols, perword_cols):
    data_counting[perword_col] = data_counting[raw_col] / data_counting['n_words']

# Z-Normalisierung der (wortzahl-normalisierten) Häufigkeitsvektoren
for col in perword_cols:
    data_counting[col + '_z'] = stats.zscore(data_counting[col])

# Z-Normalisierung der Embedding-Vektoren (entfernt den dimensionsspezifischen Bias)
embraw_cols = ['emb_care', 'emb_fairness', 'emb_loyalty', 'emb_authority', 'emb_purity']
for col in embraw_cols:
    data_embedding[col + '_z'] = stats.zscore(data_embedding[col])

# Spaltennamen für die Korrelation
mfq_cols = ['mfq_care_score', 'mfq_fairness_score', 'mfq_loyalty_score', 'mfq_authority_score', 'mfq_purity_score']
score_z_cols = [col + '_z' for col in perword_cols]
emb_cols = ['emb_care_z', 'emb_fairness_z', 'emb_loyalty_z', 'emb_authority_z', 'emb_purity_z']


# generische Funktion zur Berechnung der Korrelation zwischen zwei Vektoren.
# Ist das Rohsignal einer Person in allen Dimensionen identisch (i.d.R. ueberall 0
# Treffer), hat es selbst keine Varianz. Die z-Transformation wuerde daraus trotzdem
# einen scheinbar variierenden Vektor erzeugen, der aber nur die unterschiedlichen
# Wortlisten-Laengen je Dimension widerspiegelt, nicht die Person - die Korrelation
# waere dann ein reines Transformationsartefakt. Im Fall von Varianz 0 im Rohsignal
# ist die Korrelation entsprechend nicht definiert (NaN) statt eines solchen Kunstwerts.
def correlate_vectors(row, raw_cols, score_cols):
    if np.ptp(row[raw_cols].values.astype(float)) == 0:
        return np.nan
    mfq_vec   = row[mfq_cols].values.astype(float)
    score_vec = row[score_cols].values.astype(float)
    r, p = stats.pearsonr(mfq_vec, score_vec)
    return r

# Aufruf für counting und embedding
data_counting['pearson_r'] = data_counting.apply(lambda row: correlate_vectors(row, rawscore_cols, score_z_cols), axis=1)
data_embedding['pearson_r'] = data_embedding.apply(lambda row: correlate_vectors(row, embraw_cols, emb_cols), axis=1)

#Korrelationen mitteln (NaN-Faelle mit varianzlosem Rohsignal werden dabei automatisch
#ausgeschlossen, da pandas .mean() NaN ignoriert)
n_undefined_counting = data_counting['pearson_r'].isna().sum()
mean_r_counting = data_counting['pearson_r'].mean()
mean_r_embedding = data_embedding['pearson_r'].mean()
print(f'Wordcounting: {n_undefined_counting}/{len(data_counting)} Personen mit varianzlosem Rohsignal (Korrelation nicht definiert)')
print(f'Mean Pearson r for counting (N={data_counting["pearson_r"].notna().sum()}): {mean_r_counting}')
print(f'Mean Pearson r for embedding: {mean_r_embedding}')

#Ergebnisse speichern
data_counting.to_excel('Wordcounting_Analysis.xlsx', index=False)
data_embedding.to_excel('Embedding_Analysis.xlsx', index=False)

# ------------------------------------------------------------------
# PLOT 1: Boxplot der Verteilung der individuellen pearson_r-Werte je Methode
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.5))
box_data = [data_counting['pearson_r'].dropna(), data_embedding['pearson_r'].dropna()]
#Mean manuell eingeblendet
ax.boxplot(box_data, showmeans=True, meanline=True,
           meanprops={'color': 'red', 'linestyle': '--'},
           medianprops={'linewidth': 0}) 
ax.set_xticks([1, 2])
ax.set_xticklabels(['Wordcounting', 'Word Embedding'])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Pearson r pro Person')
ax.set_title('Streuung der individuellen Korrelationen')
 
# Mean r je Methode unterhalb der x-Achse einblenden
box_means = [d.mean() for d in box_data]
for i, m in enumerate(box_means, start=1):
    ax.text(i, -0.08, f'Mean r = {m:.3f}', ha='center', va='top',
             fontsize=10, transform=ax.get_xaxis_transform())
 
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.savefig(os.path.join(PLOT_DIR, '07_pearson_r_verteilung_boxplot.png'), dpi=150)
plt.close(fig)

print(f"\nPlots gespeichert in: ./{PLOT_DIR}/")
 