#!/bin/sh
# Launch a training command detached from any terminal or Claude session, with the
# machine kept awake while it runs. Survives the launching shell exiting.
#
#   scripts/train_bg.sh runs/joint3 --mode joint --steps 8000
#   tail -f runs/joint3/train.log        # watch
#   kill $(cat runs/joint3/pid)          # stop
#
# Interrupted runs continue from their last checkpoint with:
#   scripts/train_bg.sh runs/joint3 --continue runs/joint3/last.pt --mode joint --steps 8000
set -e
out="$1"; shift
mkdir -p "$out"
nohup sh -c "caffeinate -i uv run scripts/train_terrain.py --out '$out' $* >> '$out/train.log' 2>&1" > /dev/null 2>&1 &
echo $! > "$out/pid"
echo "started (pid $(cat "$out/pid")), log: $out/train.log"
