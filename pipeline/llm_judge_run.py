"""LLM-as-a-Judge: Texte bewerten lassen und die Rohantworten speichern.

    python -m pipeline.llm_judge_run --backend fake --unit person --runs 3 --subset dev
    python -m pipeline.llm_judge_run --backend ollama --model <name> --unit topic --runs 3
    python -m pipeline.llm_judge_run --backend anthropic --confirm-data-protection --unit person --runs 3

Der API-Bewerter sendet Teilnehmertexte an einen externen Dienst und startet nur mit
--confirm-data-protection, also erst nach geklärtem Datenschutz. --dry-run zeigt nur den Umfang.

Rohantworten werden Zeile für Zeile in results/llm_judge/ angehängt (nicht im Repo). Ein
abgebrochener Lauf setzt beim nächsten Start dort fort, wo er aufgehört hat.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src import config
from src.llm_judge import AnthropicJudge, FakeJudge, OllamaJudge, fingerprint
from src.llm_units import UNITS, build_units, select_subset


def output_path(backend, model, unit):
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return config.LLM_DIR / f"raw_{backend}_{safe_model}_{unit}.jsonl"


def load_done(path, fp):
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["fingerprint"] == fp:
                done.add((entry["unit_id"], entry["run"]))
    return done


def run_judging(units, judge, runs, path, workers=4):
    """Bewertet jede Einheit `runs`-mal; bereits vorhandene Läufe (gleiche Rubrik) werden übersprungen."""
    fp = fingerprint()
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path, fp)
    todo = [(row.unit_id, row.text, run) for row in units.itertuples() for run in range(runs)
            if (row.unit_id, run) not in done]
    lock = threading.Lock()
    failures = []

    def work(item):
        unit_id, text, run = item
        try:
            result = judge.score(text, run=run)
        except Exception as exc:  # einzelne Fehler stoppen nicht den ganzen Lauf, Wiederholung per Neustart
            failures.append((unit_id, run, repr(exc)))
            return
        entry = {"unit_id": unit_id, "run": run, "backend": judge.name, "model": judge.model,
                 "fingerprint": fp, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "scores": result.scores, "usage": result.usage, "raw": result.raw}
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    return {"neu": len(todo) - len(failures), "bereits_vorhanden": len(done), "fehler": failures}


def make_judge(args):
    if args.backend == "fake":
        return FakeJudge()
    if args.backend == "ollama":
        if not args.model:
            raise SystemExit("--model ist für den lokalen Bewerter nötig (Name des Ollama-Modells)")
        return OllamaJudge(args.model)
    return AnthropicJudge(model=args.model or "claude-opus-5-5", effort=args.effort,
                          data_protection_confirmed=args.confirm_data_protection)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["fake", "ollama", "anthropic"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--unit", choices=UNITS, default="person")
    parser.add_argument("--subset", choices=["dev", "holdout", "all"], default="dev",
                        help="dev: Entwicklungsteilmenge (Rubrik verfeinern), holdout: Rest (Validierung)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, help="nur die ersten N Einheiten (zum Testen)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--confirm-data-protection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    units = select_subset(build_units(args.unit), args.subset)
    if args.limit:
        units = units.head(args.limit)
    words = int(units["text"].str.split().str.len().sum())
    print(f"Einheit {args.unit}, Teilmenge {args.subset}: {len(units)} Texte, {words} Wörter, "
          f"{args.runs} Läufe = {len(units) * args.runs} Aufrufe. Rubrik-Fingerabdruck {fingerprint()}")
    if args.dry_run:
        print("Trockenlauf: nichts gesendet, nichts gespeichert.")
        return

    judge = make_judge(args)
    path = output_path(judge.name, judge.model, args.unit)
    summary = run_judging(units, judge, args.runs, path, workers=args.workers)
    print(f"Fertig: {summary['neu']} neue Bewertungen, {summary['bereits_vorhanden']} schon vorhanden, "
          f"{len(summary['fehler'])} Fehler -> {path}")
    for unit_id, run, err in summary["fehler"][:5]:
        print(f"  Fehler bei {unit_id} Lauf {run}: {err}")


if __name__ == "__main__":
    main()
