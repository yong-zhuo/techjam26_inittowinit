from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Runs the evaluator and appends the commit and metrics to runs.log.
RESULTS = Path("results.json")
LOG = Path("runs.log")


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    subprocess.run([sys.executable, "-m", "evaluator.local_evaluator"], check=True)

    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "commit": git_hash(),
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "score": result["recommended_technical_score"],
        "note": args.note,
    }
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"logged to {LOG}: {row}")


if __name__ == "__main__":
    main()
