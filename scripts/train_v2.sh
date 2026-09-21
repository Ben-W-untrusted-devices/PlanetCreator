#!/bin/sh
# Two-phase texture-generating recipe for the 2.4 km joint model, detached and
# self-resuming: run this script again after any interruption (sleep, reboot, a
# Claude session ending) and it continues each phase from its last checkpoint.
#
#   phase 1  runs/joint4      L1 + gradient + VGG perceptual, base=48, 10k steps
#   phase 2  runs/joint4_adv  + hinge PatchGAN (adv 0.05, D lr 1e-4), perceptual kept; 6k steps
#            (R1 was tried and dropped: at any effective strength it stops the small
#            PatchGAN learning at all — see docs/plan-resolution.md)
#
#   scripts/train_v2.sh              # start / resume, detached, machine kept awake
#   tail -f runs/joint4/train.log    # watch (then runs/joint4_adv/train.log)
#   kill $(cat runs/train_v2.pid)    # stop
set -e
cd "$(dirname "$0")/.."
P1=runs/joint4; P2=runs/joint4_adv
S1=10000; S2=6000

phase() {  # dir steps [trainer args...]
  dir=$1; steps=$2; shift 2
  if [ -f "$dir/done" ]; then echo "$dir already complete"; return; fi
  mkdir -p "$dir"
  if [ -f "$dir/last.pt" ]; then
    uv run scripts/train_terrain.py --continue "$dir/last.pt" --out "$dir" --steps "$steps" "$@"
  else
    uv run scripts/train_terrain.py --out "$dir" --steps "$steps" "$@"
  fi
  touch "$dir/done"
}

if [ "$1" = "--fg" ]; then
  phase "$P1" "$S1" --mode joint --base 48 --perc 0.1 --eval-every 1000 >> "$P1/train.log" 2>&1
  phase "$P2" "$S2" --mode joint --resume "$P1/last.pt" --base 48 --perc 0.1 --adv 0.05 --d-lr 1e-4 --lr 1e-4 --eval-every 500 >> "$P2/train.log" 2>&1
  echo "train_v2 complete: $(date)" >> "$P2/train.log"
  exit 0
fi

mkdir -p "$P1" "$P2" runs
nohup caffeinate -i sh "$0" --fg > /dev/null 2>&1 &
echo $! > runs/train_v2.pid
echo "started (pid $(cat runs/train_v2.pid)); logs: $P1/train.log then $P2/train.log"
