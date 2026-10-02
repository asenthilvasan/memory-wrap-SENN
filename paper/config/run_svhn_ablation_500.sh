#!/bin/bash
# ============================================================================
# SVHN ablation sweep at train_examples=500 (lower-budget variant of
# run_svhn_ablation.sh's 2000-sample sweep).
#
# Cells run: 1 (Scratch+Linear), 2 (Scratch+MW), 3 (SupCon+Linear frozen),
# 5 (SupCon+MW frozen). Cells 4 and 6 (fine-tune) are intentionally skipped
# to save compute; the frozen cells alone answer "does SupCon's lift over MW
# survive at a quarter of the original 2000-budget data?".
#
# Pretraining (one encoder per seed) uses batch 64 because batch 256 +
# drop_last on 500 samples gives only 1 SGD step/epoch; batch 64 gives 7,
# matching the 2000-budget recipe's 280 total updates (7 * 40). LR follows
# the linear scaling rule: 0.5 * 64 / 256 = 0.125.
#
# Extra arguments are passed to every python call, e.g. --wandb.
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd "$(dirname "$0")/.."

ENC=models/SVHN/supcon/mobilenet/500
LOG=logs/svhn_500
YAML=config/train.yaml

mkdir -p "$LOG"

# Flip yaml: dataset_name=SVHN, train_examples=500. Back up original first.
cp "$YAML" "${YAML}.bak"
sed -i 's/^dataset_name:.*/dataset_name: SVHN/' "$YAML"
sed -i 's/^train_examples:.*/train_examples: 500/' "$YAML"
sed -i 's/^batch_size_train:.*/batch_size_train: 64/' "$YAML"
echo "Set $YAML -> dataset_name: SVHN, train_examples: 500, batch_size_train: 64"
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
    --dataset=SVHN --loss=supcon --model=mobilenet \
    --train_examples=500 --epochs=40 --batch_size=64 \
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
