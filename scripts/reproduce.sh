#!/usr/bin/env bash
# Reproduce the pilot end to end: probes, role projections, generations, scores, tables + plots.
# Resumable: re-running skips finished stages (outputs cached under runs/pilot/<model>/).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pytest -q tests
python -m rpg.run configs/${1:-pilot}.yaml
