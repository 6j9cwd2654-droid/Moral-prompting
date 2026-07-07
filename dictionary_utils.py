import pandas as pd

DIMS = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']

def load_dictionaries(path='mfd_de_referenz.xlsx'):
    dict_df = pd.read_excel(path, sheet_name='MFD_DE')
    dictionaries = {}
    for dim in DIMS:
        dictionaries[dim] = dict_df[dict_df[dim] == 1]['match_pattern'].tolist()
    return dictionaries