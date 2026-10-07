#!/usr/bin/env bash
#SBATCH --job-name=p2-sst
#SBATCH --time=10:00:00
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PYTHON=${PYTHON:-python3}
CONDA_ENV=${CONDA_ENV:-dnlp}
MODE=${1:-${MODE:-sst_all}}
SST_SEEDS_3=${SST_SEEDS_3:-"11711 42 2026"}
SST_SEEDS_5=${SST_SEEDS_5:-"11711 42 2026 3407 8848"}

mkdir -p runs predictions/bert models

# Prefer node-local scratch for transient BERT ensemble checkpoints.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export SST_ENSEMBLE_TMP="${SLURM_TMPDIR}/dnlp-sst-${SLURM_JOB_ID:-job}"
else
    export SST_ENSEMBLE_TMP="${TMPDIR:-/tmp}/dnlp-sst-${USER:-user}-${SLURM_JOB_ID:-job}"
fi
mkdir -p "${SST_ENSEMBLE_TMP}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    source activate "${CONDA_ENV}"
fi

echo "============================================================"
echo "P2 SST run"
echo "mode              : ${MODE}"
echo "host              : $(hostname)"
echo "working directory : ${PWD}"
echo "python            : $(${PYTHON} --version 2>&1)"
echo "scratch           : ${SST_ENSEMBLE_TMP}"
echo "============================================================"
echo
quota -s 2>/dev/null || true
df -h "${SST_ENSEMBLE_TMP}" 2>/dev/null || true
nvidia-smi 2>/dev/null || true
echo

COMMON=(
    --option finetune
    --task sst
    --use_gpu
    --local_files_only
)

run_cmd() {
    echo
    echo "======================================================================"
    printf 'RUNNING:'
    printf ' %q' "$@"
    echo
    echo "======================================================================"
    "$@"
}

run_profile() {
    local profile="$1"
    local history="$2"
    run_cmd "${PYTHON}" -u multitask_classifier.py \
        "${COMMON[@]}" \
        --sst_profile "${profile}" \
        --sst_history_out "${history}"
}

run_baseline() {
    run_profile readme_baseline runs/sst_01_baseline.csv
}

run_dropout() {
    run_profile readme_dropout runs/sst_02_dropout.csv
}

run_low_lr() {
    run_profile readme_low_lr runs/sst_03_low_lr.csv
}

run_smoothing() {
    run_profile readme_smoothing runs/sst_04_label_smoothing.csv
}

run_plateau() {
    run_profile readme_plateau runs/sst_05_plateau.csv
}

run_ensemble3() {
    # Strong final system. Four representations x three seeds, but only a
    # controlled top-4 ensemble search is performed on DEV.
    read -r -a seeds <<< "${SST_SEEDS_3}"
    run_cmd "${PYTHON}" -u multitask_classifier.py \
        "${COMMON[@]}" \
        --sst_profile accuracy \
        --ensemble_seeds "${seeds[@]}" \
        --ensemble_heads pooler raw_cls cls_mean gated_attn \
        --ensemble_shortlist_size 4 \
        --ensemble_max_subset_size 4 \
        --ensemble_temperatures 0.9 1.0 1.1 \
        --ensemble_checkpoint_dir "${SST_ENSEMBLE_TMP}" \
        --sst_history_out runs/sst_06_accuracy_ensemble.csv
}

run_ensemble5() {
    # Higher-compute search: four representations x five seeds.
    # The best individual checkpoint is always retained as a candidate, so the
    # final DEV-selected system cannot be worse than the best trained member.
    read -r -a seeds <<< "${SST_SEEDS_5}"
    run_cmd "${PYTHON}" -u multitask_classifier.py \
        "${COMMON[@]}" \
        --sst_profile accuracy \
        --use_llrd \
        --llrd_decay 0.95 \
        --head_lr 2e-5 \
        --weight_decay 0.01 \
        --grad_clip_norm 1.0 \
        --batch_size 32 \
        --epochs 15 \
        --early_stop_patience 4 \
        --min_delta 0.0 \
        --ensemble_seeds "${seeds[@]}" \
        --ensemble_heads pooler raw_cls cls_mean gated_attn \
        --ensemble_shortlist_size 4 \
        --ensemble_max_subset_size 4 \
        --ensemble_temperatures 0.9 1.0 1.1 \
        --ensemble_checkpoint_dir "${SST_ENSEMBLE_TMP}" \
        --sst_history_out runs/sst_06_accuracy_ensemble5.csv
}

run_ordinal_probe() {
    # Separate, controlled research ablation. Do not silently mix this result
    # into E1-E5; compare it against the CE final system.
    run_cmd "${PYTHON}" -u multitask_classifier.py \
        "${COMMON[@]}" \
        --lr 1e-5 \
        --hidden_dropout_prob 0.1 \
        --use_llrd \
        --llrd_decay 0.95 \
        --head_lr 2e-5 \
        --weight_decay 0.01 \
        --grad_clip_norm 1.0 \
        --use_warmup_linear_decay \
        --warmup_ratio 0.1 \
        --use_ema \
        --ema_decay 0.995 \
        --ema_start_epoch 2 \
        --ordinal_aux_lambda 0.05 \
        --sst_head_type pooler \
        --batch_size 32 \
        --epochs 15 \
        --early_stop_patience 4 \
        --experiment_name ordinal-aux-005 \
        --sst_history_out runs/sst_ordinal_aux_005.csv
}

run_all() {
    run_baseline
    run_dropout
    run_low_lr
    run_smoothing
    run_plateau
    run_ensemble3
}

case "${MODE}" in
    sst_all)       run_all ;;
    sst_baseline)  run_baseline ;;
    sst_dropout)   run_dropout ;;
    sst_low_lr)    run_low_lr ;;
    sst_smoothing) run_smoothing ;;
    sst_plateau)   run_plateau ;;
    sst_ensemble|sst_ensemble3) run_ensemble3 ;;
    sst_ensemble5) run_ensemble5 ;;
    sst_ordinal)   run_ordinal_probe ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Valid: sst_all sst_baseline sst_dropout sst_low_lr sst_smoothing sst_plateau sst_ensemble3 sst_ensemble5 sst_ordinal" >&2
        exit 2
        ;;
esac

echo
echo "============================================================"
echo "FINISHED: ${MODE}"
echo "============================================================"
quota -s 2>/dev/null || true