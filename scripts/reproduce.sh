#!/usr/bin/env bash
# Reproduce a config end to end (default: single): probes, role projections, tables + plots.
# Resumable: re-running skips finished stages (outputs cached under runs/<config>/<model>/).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pytest -q tests
python -m rpg.run configs/${1:-single}.yaml
