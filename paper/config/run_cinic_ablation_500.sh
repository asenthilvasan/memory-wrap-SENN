#!/bin/bash
# ============================================================================
# CINIC-10 ablation sweep at train_examples=500 (lower-budget variant of
# run_cinic_ablation.sh's 2000-sample sweep). Mirrors the recipe that worked
# for run_svhn_ablation_500.sh: shrink both pretrain and downstream batch
# size to 64, apply linear LR scaling at pretrain, keep epoch counts at the
# CINIC defaults so total SGD update counts roughly match the 2000-budget
# original (7 steps/epoch * 300 epochs = 2100 updates per stage).
#
# Cells run: 1 (Scratch+Linear), 2 (Scratch+MW), 3 (SupCon+Linear frozen),
# 5 (SupCon+MW frozen). Cells 4 and 6 (fine-tune) are intentionally skipped
# to save compute, matching the SVHN-500 sweep.
#
# PREREQUISITE: stage the dataset once with `bash config/setup_cinic.sh`.
#
# Pretrains one SupCon encoder per seed first (batch 64, lr 0.5 * 64 / 256 =
# 0.125). Sanity-check the [diag ep1/batch1] line in each pretrain log:
# per_dim_std should be well above 0 and mean_cos well below 1.0. If you see
# collapse, retry with --projection_bn=True or a smaller --lr.
#
# WARNING: at 300 epochs per stage this sweep is much longer than the SVHN
#   40-epoch version, and pretraining now runs once per seed. Run inside
#   tmux. Extra arguments are passed to every python call, e.g. --wandb.
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd "$(dirname "$0")/.."

ENC=models/CINIC10/supcon/mobilenet/500
LOG=logs/cinic_500
YAML=config/train.yaml

mkdir -p "$LOG"

# Verify dataset is staged (setup_cinic.sh creates this symlink).
if [ ! -d datasets/CINIC10/train ]; then
    echo "ERROR: datasets/CINIC10/train not found." >&2
    echo "Run config/setup_cinic.sh first." >&2
    exit 1
fi

# Flip yaml: dataset_name=CINIC10, train_examples=500, batch_size_train=64.
# Back up original first; restore_yaml below puts everything back on exit.
cp "$YAML" "${YAML}.bak"
sed -i 's/^dataset_name:.*/dataset_name: CINIC10/' "$YAML"
sed -i 's/^train_examples:.*/train_examples: 500/' "$YAML"
sed -i 's/^batch_size_train:.*/batch_size_train: 64/' "$YAML"
echo "Set $YAML -> dataset_name: CINIC10, train_examples: 500, batch_size_train: 64"
grep -E "^(dataset_name|train_examples|batch_size_train):" "$YAML"

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
    --train_examples=500 --epochs=300 --batch_size=64 \
    --lr=0.125 --temperature=0.07 --projection_dim=0 "$@"

echo "===== Cell 1: Scratch + Linear ====="
python -u train.py --modality=std "$@" \
    2>&1 | tee $LOG/01_scratch_linear.txt

echo "===== Cell 2: Scratch + MW ====="
python -u train.py --modality=encoder_memory "$@" \
    2>&1 | tee $LOG/02_scratch_mw.txt

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/03_supcon_linear_frozen.txt

# Cell 4 (SupCon + Linear fine-tune) intentionally skipped.

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True "$@" \
    2>&1 | tee $LOG/05_supcon_mw_frozen.txt

# Cell 6 (SupCon + MW fine-tune) intentionally skipped.

echo "===== ALL DONE ====="
