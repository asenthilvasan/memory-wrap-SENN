#!/bin/bash
# Pretrain one encoder per downstream seed (0..RUNS-1), so each train.py run
# loads an encoder that only saw that run's training subset. Seeds that
# already have a checkpoint are skipped, so a crashed sweep can be resumed.
#
# Usage (from paper/):
#   bash config/pretrain_encoders.sh ENC_DIR RUNS LOG_DIR [pretrain_supcon.py flags...]
#
# ENC_DIR must be the directory pretrain_supcon.py saves to for those flags,
# e.g. models/SVHN/supcon/mobilenet/2000.
set -e
set -o pipefail

ENC_DIR=$1
RUNS=$2
LOG=$3
shift 3

mkdir -p "$LOG"
for SEED in $(seq 0 $((RUNS - 1))); do
    if [ -f "$ENC_DIR/seed$SEED.pt" ]; then
        echo "Encoder for seed $SEED exists, skipping."
        continue
    fi
    echo "===== Pretraining encoder for seed $SEED ====="
    python -u pretrain_supcon.py --seed="$SEED" "$@" \
        2>&1 | tee "$LOG/00_pretrain_seed$SEED.txt"
    if [ ! -f "$ENC_DIR/seed$SEED.pt" ]; then
        echo "ERROR: expected $ENC_DIR/seed$SEED.pt after pretraining. Do the flags match ENC_DIR?" >&2
        exit 1
    fi
done
