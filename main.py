"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA 6 trophy's Exophase
earn rate at the case's target as five quantiles, scored by pinball loss on
the logit scale.

Reads every <id>.in in the input directory (the grader runs one case at a
time, with only that case's file there) and writes <id>.out to the output
directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one
row per trophy, each q five finite numbers from 0 to 100, non-decreasing.

The model lives in model.py: per-trophy logit-linear growth on ln(1 + hours
since launch), shrunk toward the tier median slope, anchored on the last
observed rate, with an honest spread (random walk + fit variance + tier-borrow
uncertainty + a floor). With a flat history it degrades exactly to the
persistence starter.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts it
back first so `import model` finds model.py shipped beside this file.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import os
import sys
from pathlib import Path

# The grader's -P leaves this directory OFF sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        # Write atomically (temp file + rename) so an interrupted run never
        # leaves a half-written .out behind for the grader to read.
        target = OUTPUT_DIR / f"{path.stem}.out"
        tmp = target.with_suffix(".out.tmp")
        tmp.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")
        tmp.replace(target)


if __name__ == "__main__":
    main()
