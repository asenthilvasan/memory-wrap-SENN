#!/bin/bash
# ============================================================================
# CINIC-10 ablation sweep, mirrors run_svhn_ablation.sh
#   6 cells: scratch/supcon x linear/MW x frozen/finetune
#   Uses 300 epochs for both pretrain and downstream (per CINIC10 block in
#   config/train.yaml; pretrain epochs passed via CLI).
#
# PREREQUISITE: stage the dataset once with `bash config/setup_cinic.sh`.
#
# Pretrains one SupCon encoder per seed first, using the feature-space recipe
# (no projection head). For the canonical recipe swap in:
#   --projection_dim=128 --projection_bn=True --warmup_epochs=10 --lr=0.1
#
# WARNING: at 300 epochs and 15 runs per cell, this sweep takes considerably
#   longer than the SVHN 40-epoch version, and pretraining now runs once per
#   seed. Run inside tmux. Extra arguments are passed to every python call,
#   e.g. --wandb.
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd "$(dirname "$0")/.."

ENC=models/CINIC10/supcon/mobilenet/2000
LOG=logs/cinic_2000
YAML=config/train.yaml

mkdir -p "$LOG"

# Flip yaml dataset_name to CINIC10 (back up original so we can restore on exit)
cp "$YAML" "${YAML}.bak"
sed -i 's/^dataset_name:.*/dataset_name: CINIC10/' "$YAML"
echo "Set $YAML -> dataset_name: CINIC10"
grep "^dataset_name:" "$YAML"

# Restore yaml on exit (success, failure, or Ctrl-C)
restore_yaml() {
    if [ -f "${YAML}.bak" ]; then
        mv "${YAML}.bak" "$YAML"
        echo "Restored $YAML from backup."
    fi
}
trap restore_yaml EXIT

RUNS=$(awk '/^runs:/ {print $2}' "$YAML")
bash config/pretrain_encoders.sh "$ENC" "$RUNS" "$LOG" \
    --dataset=CINIC10 --loss=supcon --model=mobilenet \
    --train_examples=2000 --epochs=300 \
    --projection_dim=0 --temperature=0.07 --lr=0.5 "$@"

echo "===== Cell 1: Scratch + Linear ====="
python -u train.py --modality=std "$@" \
    2>&1 | tee $LOG/01_cinic_scratch_linear.txt

echo "===== Cell 2: Scratch + MW ====="
python -u train.py --modality=encoder_memory "$@" \
    2>&1 | tee $LOG/02_cinic_scratch_mw.txt

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/03_cinic_supcon_linear_frozen.txt

echo "===== Cell 4: SupCon + Linear (fine-tune) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC "$@" \
    2>&1 | tee $LOG/04_cinic_supcon_linear_finetune.txt

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/05_cinic_supcon_mw_frozen.txt

echo "===== Cell 6: SupCon + MW (fine-tune) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC "$@" \
    2>&1 | tee $LOG/06_cinic_supcon_mw_finetune.txt

echo "===== ALL DONE ====="

# ----------------------------------------------------------------------------
# Optional: auto-stop the RunPod pod when the sweep finishes, so you stop
# being billed for idle GPU time while you sleep. Uncomment if desired.
#
# runpodctl stop pod "$RUNPOD_POD_ID"
# ----------------------------------------------------------------------------
