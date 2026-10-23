import pandas as pd

from src import config
from src.config import DIMS


def load_dictionaries(path=config.MFD_PATH):
    dict_df = pd.read_excel(path, sheet_name=config.MFD_SHEET)
    return {dim: dict_df[dict_df[dim] == 1]["match_pattern"].tolist() for dim in DIMS}
