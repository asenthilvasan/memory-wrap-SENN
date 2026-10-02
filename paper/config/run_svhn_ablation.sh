#!/bin/bash
# ============================================================================
# SVHN ablation sweep at train_examples=2000 (config/train.yaml defaults).
#   6 cells: scratch/supcon x linear/MW x frozen/finetune
#
# Pretrains one SupCon encoder per seed first (40 epochs, batch 256), then
# runs the downstream cells. Extra arguments are passed to every python call,
# e.g. `bash config/run_svhn_ablation.sh --wandb`.
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd "$(dirname "$0")/.."

ENC=models/SVHN/supcon/mobilenet/2000
LOG=logs/svhn_2000
RUNS=$(awk '/^runs:/ {print $2}' config/train.yaml)

mkdir -p "$LOG"

bash config/pretrain_encoders.sh "$ENC" "$RUNS" "$LOG" \
    --dataset=SVHN --loss=supcon --model=mobilenet \
    --train_examples=2000 --epochs=40 --batch_size=256 \
    --lr=0.5 --temperature=0.07 --projection_dim=0 "$@"

echo "===== Cell 1: Scratch + Linear ====="
python -u train.py --modality=std "$@" \
    2>&1 | tee $LOG/01_scratch_linear.txt

echo "===== Cell 2: Scratch + MW ====="
python -u train.py --modality=encoder_memory "$@" \
    2>&1 | tee $LOG/02_scratch_mw.txt

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/03_supcon_linear_frozen.txt

echo "===== Cell 4: SupCon + Linear (fine-tune) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC "$@" \
    2>&1 | tee $LOG/04_supcon_linear_finetune.txt

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/05_supcon_mw_frozen.txt

echo "===== Cell 6: SupCon + MW (fine-tune) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC "$@" \
    2>&1 | tee $LOG/06_supcon_mw_finetune.txt

echo "===== ALL DONE ====="
