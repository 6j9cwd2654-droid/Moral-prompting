"""Instrumentvalidierung mit den Bewertungen des LLM-Bewerters.

    python -m pipeline.llm_validation --backend fake --model fake --unit person --subset holdout

1. Verlässlichkeit: Übereinstimmung der Läufe (ICC) je Dimension
2. Konvergente Validität: Spearman-Korrelationen zwischen Bewerter-Scores und den Wordcounting-/
   Embedding-Scores je Text; passende Dimensionen (Diagonale) sollten stärker korrelieren als
   unpassende (Multitrait-Multimethod-Logik)
3. Hypothesentest mit Bewerter-Scores statt Wörterbuch-Scores (nur Einheit 'person'):
   Profilkorrelation, Dimensionseffekte und RSA

Läuft die Auswertung auf Ausgaben des fake-Bewerters, dienen die Zahlen nur der Funktionsprüfung.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.config import DIMS, MFQ_COLS
from src.data_utils import load_mfq
from src.llm_judge import DIM_KEYS, fingerprint
from src.llm_units import UNITS, build_units, select_subset
from src.reliability import icc_two_way
from src.rsa_utils import zscore_columns
from src.stats_utils import profile_correlation
from pipeline.llm_judge_run import output_path
from pipeline.rsa_analysis import run_rsa


def load_runs(path, fp):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["fingerprint"] == fp:
            rows.append({"unit_id": entry["unit_id"], "run": entry["run"], **entry["scores"]})
    if not rows:
        raise SystemExit(f"Keine Bewertungen mit Rubrik-Fingerabdruck {fp} in {path}")
    return pd.DataFrame(rows)


def reliability(runs):
    rows = []
    for key in DIM_KEYS:
        wide = runs.pivot(index="unit_id", columns="run", values=key).dropna()
        single, mean = icc_two_way(wide.values) if wide.shape[1] > 1 else (np.nan, np.nan)
        rows.append({"dimension": key, "n_texte": len(wide), "n_laeufe": wide.shape[1],
                     "icc_einzel": single, "icc_mittel": mean, "anteil_nullen": float((wide.values == 0).mean())})
    return pd.DataFrame(rows)


def instrument_scores(unit):
    """Wordcounting-Rate und Embedding-Score je Bewertungseinheit (person oder topic)."""
    scoring = pd.read_excel(config.SCORING_PATH)
    emb = pd.read_excel(config.EMBEDDING_SCORES_PATH)
    if unit == "person":
        table = pd.DataFrame({"unit_id": [f"p{p}" for p in scoring["participant_id"]], "topic": "alle"})
        for d in DIM_KEYS:
            table[f"wc_{d}"] = (scoring[f"score_{d}"] / scoring["n_words"]).values
            table[f"emb_{d}"] = emb[f"emb_{d}"].values
        return table
    scoring_t = pd.read_excel(config.SCORING_PERTOPIC_PATH)
    emb_t = pd.read_excel(config.PERTOPIC_ANALYSIS_PATH)
    parts = []
    for topic in config.TOPICS:
        part = pd.DataFrame({"unit_id": [f"p{p}_{topic}" for p in scoring_t["participant_id"]], "topic": topic})
        for d in DIM_KEYS:
            part[f"wc_{d}"] = (scoring_t[f"{topic}_score_{d}"] / scoring_t[f"{topic}_n_words"]).values
            part[f"emb_{d}"] = emb_t[f"emb_{topic}_{d}"].values
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def convergence(judge_median, instruments):
    """5x5-Spearman-Matrizen Bewerter x Instrument; je Thema getrennt (Einheiten derselben Person sind abhängig)."""
    merged = judge_median.merge(instruments, on="unit_id")
    results = []
    for topic, group in merged.groupby("topic"):
        for prefix, label in (("wc", "Wordcounting"), ("emb", "Embedding")):
            matrix = pd.DataFrame(index=DIM_KEYS, columns=DIM_KEYS, dtype=float)
            for j in DIM_KEYS:
                for i in DIM_KEYS:
                    judged, measured = group[f"judge_{j}"], group[f"{prefix}_{i}"]
                    matrix.loc[j, i] = (np.nan if judged.nunique() < 2 or measured.nunique() < 2
                                        else stats.spearmanr(judged, measured)[0])
            diag = np.diag(matrix.values)
            off = matrix.values[~np.eye(len(DIM_KEYS), dtype=bool)]
            results.append({"thema": topic, "instrument": label, "n": len(group),
                            "mittel_diagonale": np.nanmean(diag), "mittel_ausserhalb": np.nanmean(off),
                            "differenz": np.nanmean(diag) - np.nanmean(off), "matrix": matrix})
    return results


def hypothesis_tests(judge_median):
    """Dieselben Analysen wie mit den Wörterbuch-Scores, aber mit den Bewerter-Scores (Einheit person)."""
    mfq = load_mfq()
    pids = [int(u[1:]) for u in judge_median["unit_id"]]
    data = judge_median.assign(participant_id=pids).merge(mfq, on="participant_id")
    raw = data[[f"judge_{d}" for d in DIM_KEYS]].values.astype(float)
    mfq_raw = data[MFQ_COLS].values.astype(float)
    informative = [k for k in range(len(DIM_KEYS)) if raw[:, k].std() > 0]
    if len(informative) < len(DIM_KEYS):
        print("Hinweis: Dimensionen ohne Varianz in den Bewerter-Scores:",
              [DIM_KEYS[k] for k in range(len(DIM_KEYS)) if k not in informative])
    z = np.zeros_like(raw)
    z[:, informative] = zscore_columns(raw[:, informative])
    r = [profile_correlation(raw[i], z[i], mfq_raw[i]) for i in range(len(raw))]
    rows = [{"test": "Profilkorrelation je Person, Mittelwert", "wert": float(np.nanmean(r)),
             "n": int(np.sum(~np.isnan(r))), "p_einseitig": np.nan}]
    for k in informative:
        rr, pp = stats.pearsonr(mfq_raw[:, k], raw[:, k])
        rows.append({"test": f"Dimensionseffekt {DIMS[k]} (Pearson r)", "wert": rr, "n": len(raw), "p_einseitig": np.nan})
    if len(informative) == len(DIM_KEYS):
        res = run_rsa(mfq_raw, raw)
        rows.append({"test": "RSA gesamt (Mantel-r)", "wert": res["r"], "n": res["n"], "p_einseitig": res["p_one_sided"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--unit", choices=["person", "topic"], default="person")
    parser.add_argument("--subset", choices=["dev", "holdout", "all"], default="holdout")
    args = parser.parse_args()

    runs = load_runs(output_path(args.backend, args.model, args.unit), fingerprint())
    allowed = set(select_subset(build_units(args.unit), args.subset)["unit_id"])
    runs = runs[runs["unit_id"].isin(allowed)]
    print(f"{runs['unit_id'].nunique()} Texte, {runs['run'].nunique()} Läufe (Teilmenge {args.subset})")

    rel = reliability(runs)
    median = runs.groupby("unit_id")[DIM_KEYS].median().add_prefix("judge_").reset_index()
    conv = convergence(median, instrument_scores(args.unit))

    pd.set_option("display.width", 200)
    print("\n--- Verlässlichkeit zwischen den Läufen ---")
    print(rel.round(3).to_string(index=False))
    print("\n--- Konvergente Validität (Spearman, Diagonale = passende Dimension) ---")
    summary = pd.DataFrame([{k: v for k, v in c.items() if k != "matrix"} for c in conv])
    print(summary.round(3).to_string(index=False))

    with pd.ExcelWriter(config.LLM_VALIDATION_PATH) as writer:
        rel.to_excel(writer, sheet_name="Verlaesslichkeit", index=False)
        summary.to_excel(writer, sheet_name="Konvergenz_Uebersicht", index=False)
        for c in conv:
            c["matrix"].to_excel(writer, sheet_name=f"Konv_{c['instrument'][:4]}_{c['thema'][:12]}")
        if args.unit == "person":
            tests = hypothesis_tests(median)
            tests.to_excel(writer, sheet_name="Hypothesentest", index=False)
            print("\n--- Hypothesentest mit Bewerter-Scores ---")
            print(tests.round(4).to_string(index=False))
    print(f"\nGespeichert: {config.LLM_VALIDATION_PATH.name}")


if __name__ == "__main__":
    main()
