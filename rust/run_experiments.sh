#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TTT Experiment Battery
# =============================================================================
# Configurable parameters - adjust these for your hardware/time budget

BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-256}"
EPOCHS="${EPOCHS:-10}"
SAMPLES="${SAMPLES:-10000}"
TEST_SAMPLES="${TEST_SAMPLES:-1000}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
LR="${LR:-5e-4}"
SIZE="${SIZE:-60m}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:42}"

# Base output directory
OUT_BASE="${OUT_BASE:-./artifacts-experiments}"

# Dry run mode - set to "true" to just print commands without running
DRY_RUN="${DRY_RUN:-false}"

# =============================================================================
# Helper function
# =============================================================================

run_experiment() {
    local name="$1"
    local inner="$2"
    local ttt_base_lr="$3"
    local mlp_expansion="${4:-4}"
    local extra_args="${5:-}"

    local out_dir="${OUT_BASE}/${name}"

    echo "=========================================="
    echo "Running: $name"
    echo "  inner=$inner, ttt_base_lr=$ttt_base_lr, mlp_expansion=$mlp_expansion"
    echo "  output: $out_dir"
    echo "=========================================="

    local cmd="cargo run --release -- train \
        --inner $inner \
        --size $SIZE \
        --batch $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --epochs $EPOCHS \
        --samples $SAMPLES \
        --test-samples $TEST_SAMPLES \
        --mini-batch-size $MINI_BATCH_SIZE \
        --lr $LR \
        --ttt-base-lr $ttt_base_lr \
        --mlp-expansion $mlp_expansion \
        --grad-accum $GRAD_ACCUM \
        --workers $WORKERS \
        --seed $SEED \
        --out $out_dir \
        $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $cmd"
        echo ""
    else
        eval "$cmd"
        echo ""
        echo "Completed: $name"
        echo ""
    fi
}

# =============================================================================
# Phase 1: LR Sweep for MLP variants (short runs to find stable LRs)
# =============================================================================

phase1_lr_sweep() {
    echo ""
    echo "############################################################"
    echo "# PHASE 1: TTT Base LR Sweep for MLP Variants"
    echo "############################################################"
    echo ""

    # mlp with expansion=4 (0.1 is known good, but test others)
    run_experiment "phase1/mlp-exp4-tttlr0.1" mlp 0.1 4
    run_experiment "phase1/mlp-exp4-tttlr0.01" mlp 0.01 4

    # mlp2 with expansion=4
    run_experiment "phase1/mlp2-exp4-tttlr0.1" mlp2 0.1 4
    run_experiment "phase1/mlp2-exp4-tttlr0.01" mlp2 0.01 4

    # mlp4 with expansion=4
    run_experiment "phase1/mlp4-exp4-tttlr0.1" mlp4 0.1 4
    run_experiment "phase1/mlp4-exp4-tttlr0.01" mlp4 0.01 4
    run_experiment "phase1/mlp4-exp4-tttlr0.001" mlp4 0.001 4

    # mlp with expansion=8
    run_experiment "phase1/mlp-exp8-tttlr0.1" mlp 0.1 8
    run_experiment "phase1/mlp-exp8-tttlr0.01" mlp 0.01 8

    # mlp2 with expansion=8
    run_experiment "phase1/mlp2-exp8-tttlr0.01" mlp2 0.01 8
    run_experiment "phase1/mlp2-exp8-tttlr0.001" mlp2 0.001 8
}

# =============================================================================
# Phase 2: Fair Comparison (use best LRs from Phase 1)
# =============================================================================
# After running Phase 1, update these LRs based on results

# Default LRs (update after Phase 1)
TTTLR_LINEAR="${TTTLR_LINEAR:-1.0}"
TTTLR_LINEAR_ADAM="${TTTLR_LINEAR_ADAM:-1.0}"
TTTLR_MLP="${TTTLR_MLP:-0.1}"
TTTLR_MLP2="${TTTLR_MLP2:-0.1}"      # Update after phase 1
TTTLR_MLP4="${TTTLR_MLP4:-0.01}"     # Update after phase 1
TTTLR_MLP_EXP8="${TTTLR_MLP_EXP8:-0.01}"  # Update after phase 1

phase2_comparison() {
    echo ""
    echo "############################################################"
    echo "# PHASE 2: Fair Model Comparison (with tuned LRs)"
    echo "############################################################"
    echo ""

    # Baseline
    run_experiment "phase2/linear-baseline" linear "$TTTLR_LINEAR" 4

    # Linear variant
    run_experiment "phase2/linear-adam" linear-adam "$TTTLR_LINEAR_ADAM" 4

    # MLP variants (expansion=4)
    run_experiment "phase2/mlp-exp4" mlp "$TTTLR_MLP" 4
    run_experiment "phase2/mlp2-exp4" mlp2 "$TTTLR_MLP2" 4
    run_experiment "phase2/mlp4-exp4" mlp4 "$TTTLR_MLP4" 4

    # MLP expansion factor comparison
    run_experiment "phase2/mlp-exp1" mlp "$TTTLR_MLP" 1
    run_experiment "phase2/mlp-exp2" mlp "$TTTLR_MLP" 2
    run_experiment "phase2/mlp-exp8" mlp "$TTTLR_MLP_EXP8" 8
}

# =============================================================================
# Phase 3: Position Encoding (on linear baseline)
# =============================================================================
# NOTE: Requires --pos-encoding flag to be added to CLI

phase3_position_encoding() {
    echo ""
    echo "############################################################"
    echo "# PHASE 3: Position Encoding Comparison"
    echo "# NOTE: Requires --pos-encoding CLI flag"
    echo "############################################################"
    echo ""

    # Uncomment when --pos-encoding is available:
    # run_experiment "phase3/linear-rope" linear "$TTTLR_LINEAR" 4 "--pos-encoding rope"
    # run_experiment "phase3/linear-none" linear "$TTTLR_LINEAR" 4 "--pos-encoding none"
    # run_experiment "phase3/linear-absolute" linear "$TTTLR_LINEAR" 4 "--pos-encoding absolute"

    echo "Position encoding experiments skipped - add --pos-encoding flag to CLI first"
}

# =============================================================================
# Phase 4: Scale comparison (run with best config from Phase 2)
# =============================================================================

BEST_INNER="${BEST_INNER:-linear}"
BEST_TTTLR="${BEST_TTTLR:-1.0}"
BEST_EXP="${BEST_EXP:-4}"

phase4_scale() {
    echo ""
    echo "############################################################"
    echo "# PHASE 4: Scale Comparison"
    echo "# Using: inner=$BEST_INNER, ttt_base_lr=$BEST_TTTLR"
    echo "############################################################"
    echo ""

    local orig_size="$SIZE"

    SIZE="12m"
    run_experiment "phase4/size-12m" "$BEST_INNER" "$BEST_TTTLR" "$BEST_EXP"

    SIZE="60m"
    run_experiment "phase4/size-60m" "$BEST_INNER" "$BEST_TTTLR" "$BEST_EXP"

    SIZE="125m"
    run_experiment "phase4/size-125m" "$BEST_INNER" "$BEST_TTTLR" "$BEST_EXP"

    SIZE="$orig_size"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [phase1|phase2|phase3|phase4|all]"
    echo ""
    echo "Environment variables for configuration:"
    echo "  BATCH_SIZE=$BATCH_SIZE"
    echo "  SEQ_LEN=$SEQ_LEN"
    echo "  EPOCHS=$EPOCHS"
    echo "  SAMPLES=$SAMPLES"
    echo "  TEST_SAMPLES=$TEST_SAMPLES"
    echo "  MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
    echo "  LR=$LR"
    echo "  SIZE=$SIZE"
    echo "  GRAD_ACCUM=$GRAD_ACCUM"
    echo "  WORKERS=$WORKERS"
    echo "  OUT_BASE=$OUT_BASE"
    echo "  DRY_RUN=$DRY_RUN"
    echo ""
    echo "Phase 2 tuned LRs (update after Phase 1):"
    echo "  TTTLR_LINEAR=$TTTLR_LINEAR"
    echo "  TTTLR_LINEAR_ADAM=$TTTLR_LINEAR_ADAM"
    echo "  TTTLR_MLP=$TTTLR_MLP"
    echo "  TTTLR_MLP2=$TTTLR_MLP2"
    echo "  TTTLR_MLP4=$TTTLR_MLP4"
    echo "  TTTLR_MLP_EXP8=$TTTLR_MLP_EXP8"
    echo ""
    echo "Phase 4 best config (update after Phase 2):"
    echo "  BEST_INNER=$BEST_INNER"
    echo "  BEST_TTTLR=$BEST_TTTLR"
    echo "  BEST_EXP=$BEST_EXP"
}

case "${1:-all}" in
    phase1)
        phase1_lr_sweep
        ;;
    phase2)
        phase2_comparison
        ;;
    phase3)
        phase3_position_encoding
        ;;
    phase4)
        phase4_scale
        ;;
    all)
        phase1_lr_sweep
        phase2_comparison
        phase3_position_encoding
        phase4_scale
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown phase: $1"
        usage
        exit 1
        ;;
esac

echo ""
echo "Done!"
