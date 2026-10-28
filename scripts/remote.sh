#!/usr/bin/env bash
# Sync this repo to a GPU box and run a config there in the background (nohup), or pull results back.
#   SSH="ssh -p 12345 root@1.2.3.4" scripts/remote.sh push
#   SSH="ssh -p 12345 root@1.2.3.4" scripts/remote.sh run pilot [extra rpg.run args]
#   SSH="ssh -p 12345 root@1.2.3.4" scripts/remote.sh log
#   SSH="ssh -p 12345 root@1.2.3.4" scripts/remote.sh pull
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SSH:?set SSH='ssh -p PORT user@host'}"
DEST=${DEST:-/workspace/hermes-role-confusion}
cmd=${1:?push|setup|run|log|pull}; shift || true
case $cmd in
  push)  rsync -az --exclude .venv --exclude external --exclude runs --exclude .git -e "${SSH% *}" ./ "${SSH##* }:$DEST/" ;;
  setup) $SSH "cd $DEST && bash scripts/setup.sh" ;;
  run)   cfg=${1:-pilot}; shift || true
         $SSH "cd $DEST && source .venv/bin/activate && export HF_HOME=/workspace/hf && \
               nohup python -m rpg.run configs/$cfg.yaml $* > run_$cfg.log 2>&1 & echo started" ;;
  log)   $SSH "tail -n 30 $DEST/run_${1:-pilot}.log" ;;
  pull)  rsync -az --exclude 'injection_acts_*.pt' -e "${SSH% *}" "${SSH##* }:$DEST/runs/" runs/ ;;
esac
