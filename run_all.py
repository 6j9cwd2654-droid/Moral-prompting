"""Führt die komplette Analyse-Pipeline in der richtigen Reihenfolge aus.

    python run_all.py                       # Hauptpipeline
    python run_all.py --with-diagnostics    # zusätzlich Deskriptives, Coverage, Sanity-Check
"""

import argparse
import subprocess
import sys

from src import config

PIPELINE = ["pipeline.wordcount", "pipeline.wordembedding", "pipeline.analysis"]
PERTOPIC = "pipeline.wordembedding_pertopic"
DIAGNOSTICS = ["diagnostics.descriptive_analysis", "diagnostics.embedding_coverage",
               "diagnostics.embedding_sanity_check"]


def run(module):
    print(f"\n=== {module} ===", flush=True)
    subprocess.run([sys.executable, "-m", module], cwd=config.ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-diagnostics", action="store_true")
    args = parser.parse_args()

    for module in PIPELINE:
        run(module)

    if config.ROHDATEN_PATH.exists():
        run(PERTOPIC)
    else:
        print(f"\n{PERTOPIC} übersprungen: Rohdaten nicht gefunden ({config.ROHDATEN_PATH})")

    if args.with_diagnostics:
        for module in DIAGNOSTICS:
            run(module)


if __name__ == "__main__":
    main()
