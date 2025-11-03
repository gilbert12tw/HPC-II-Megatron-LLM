#!/bin/bash
################################################################################
# Configuration for Megatron-LM Llama2 7B Pretraining
# Simplified parallelism and training configuration
################################################################################

# ============================================================================
# CONTAINER & PATHS
# ============================================================================
export SINGULARITY_IMAGE="/work/u4876763/megatron-nv.sif"
export MEGATRON_PATH="./Megatron-LM"
export TOKENIZER_PATH="./llama2_tokenizer"

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
export TOTAL_GPUS=8
export GPUS_PER_NODE=8
export NNODES=1

# ============================================================================
# MODEL ARCHITECTURE (Llama 2 7B - Fixed)
# ============================================================================
export NUM_LAYERS=32
export HIDDEN_SIZE=4096
export FFN_HIDDEN_SIZE=11008
export NUM_ATTENTION_HEADS=32
export VOCAB_SIZE=32000
export MAX_POSITION_EMBEDDINGS=4096

# ============================================================================
# PARALLELISM CONFIGURATION
# ============================================================================
# Default: Safe configuration that works on 8 GPUs
export TP_SIZE=2              # Tensor Parallelism (splits attention heads)
export PP_SIZE=2              # Pipeline Parallelism (splits layers)
# DP_SIZE will be auto-calculated: TOTAL_GPUS / (TP_SIZE * PP_SIZE)

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
export MICRO_BATCH_SIZE=1     # Per-GPU micro batch size
export GLOBAL_BATCH_SIZE=4    # Total batch size (should be divisible by DP_SIZE)
export SEQ_LENGTH=352         # Sequence length (must be divisible by TP when TP > 1)
export TRAIN_ITERS=100        # Number of training iterations
export LR_WARMUP_ITERS=30     # Learning rate warmup steps
export LEARNING_RATE=0.0001   # Learning rate
export MIN_LR=0.00001         # Minimum learning rate
export LR_DECAY_STYLE="cosine"  # Learning rate decay schedule

# ============================================================================
# OPTIMIZER SETTINGS
# ============================================================================
export ADAM_BETA1=0.9
export ADAM_BETA2=0.95
export WEIGHT_DECAY=0.1
export CLIP_GRAD=1.0

# ============================================================================
# PRECISION & MEMORY OPTIMIZATION
# ============================================================================
export USE_FP16=true          # V100 supports FP16 (not BF16)
export RECOMPUTE_ACTIVATIONS=true  # Gradient checkpointing to save memory
export SEQUENCE_PARALLEL=false  # Can be enabled with TP > 1
export USE_DISTRIBUTED_OPTIMIZER=true

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
export USE_MOCK_DATA=true     # Use synthetic NLTK data
export DATA_CACHE_PATH="./data_cache"

# ============================================================================
# LOGGING & CHECKPOINTING (disabled for throughput/memory testing)
# ============================================================================
export LOG_INTERVAL=10        # Log every N iterations
# export SAVE_INTERVAL=1000     # Save checkpoint every N iterations (disabled)
# export EVAL_INTERVAL=100      # Evaluation (disabled)
# export EVAL_ITERS=10          # Evaluation iterations (disabled)

# ============================================================================
# DERIVED CONFIGURATIONS (Auto-calculated)
# ============================================================================

# Calculate DP size
DP_SIZE=$((TOTAL_GPUS / (TP_SIZE * PP_SIZE)))
export DP_SIZE

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

validate_parallelism_config() {
    local tp=$1
    local pp=$2
    local num_layers=$3
    local num_heads=$4
    local total_gpus=$5
    local seq_len=$6

    local errors=0

    # Check TP divisibility with attention heads
    if [ $((num_heads % tp)) -ne 0 ]; then
        echo "ERROR: Number of attention heads ($num_heads) is not divisible by TP size ($tp)"
        echo "  Valid TP values: $(get_valid_tp_values $num_heads)"
        errors=$((errors + 1))
    fi

    # Check PP divisibility with layers
    if [ $((num_layers % pp)) -ne 0 ]; then
        echo "ERROR: Number of layers ($num_layers) is not divisible by PP size ($pp)"
        echo "  Valid PP values: $(get_valid_pp_values $num_layers)"
        errors=$((errors + 1))
    fi

    # Check TP * PP doesn't exceed total GPUs
    if [ $((tp * pp)) -gt $total_gpus ]; then
        echo "ERROR: TP ($tp) × PP ($pp) = $((tp * pp)) exceeds total GPUs ($total_gpus)"
        errors=$((errors + 1))
    fi

    # Check DP is valid (should be > 0)
    local dp=$((total_gpus / (tp * pp)))
    if [ $dp -eq 0 ]; then
        echo "ERROR: Invalid configuration results in DP=0"
        errors=$((errors + 1))
    fi

    # Check sequence length divisibility by TP (if TP > 1)
    if [ $tp -gt 1 ]; then
        if [ $((seq_len % tp)) -ne 0 ]; then
            local adjusted_seq_len=$((seq_len / tp * tp))
            echo "WARNING: Sequence length ($seq_len) is not divisible by TP ($tp)"
            echo "  Auto-adjusting to: $adjusted_seq_len"
        fi
    fi

    return $errors
}

get_valid_tp_values() {
    local num_heads=$1
    local valid=""
    for i in $(seq 1 $num_heads); do
        if [ $((num_heads % i)) -eq 0 ]; then
            valid="$valid $i"
        fi
    done
    echo "$valid"
}

get_valid_pp_values() {
    local num_layers=$1
    local valid=""
    for i in $(seq 1 $num_layers); do
        if [ $((num_layers % i)) -eq 0 ]; then
            valid="$valid $i"
        fi
    done
    echo "$valid"
}

# ============================================================================
# CONFIGURATION OVERRIDE FUNCTION
# ============================================================================

apply_config_overrides() {
    # Parse arguments and override defaults
    # Format: apply_config_overrides --tp 4 --pp 1 --micro-batch 2 ...

    while [[ $# -gt 0 ]]; do
        case $1 in
            --tp)
                TP_SIZE="$2"
                shift 2
                ;;
            --pp)
                PP_SIZE="$2"
                shift 2
                ;;
            --dp)
                # If user specifies DP explicitly, validate it's valid
                echo "INFO: DP is auto-calculated from TP and PP. Ignoring --dp $2"
                shift 2
                ;;
            --micro-batch)
                MICRO_BATCH_SIZE="$2"
                shift 2
                ;;
            --global-batch)
                GLOBAL_BATCH_SIZE="$2"
                shift 2
                ;;
            --seq-len)
                SEQ_LENGTH="$2"
                shift 2
                ;;
            --train-iters)
                TRAIN_ITERS="$2"
                shift 2
                ;;
            --lr)
                LEARNING_RATE="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                shift
                ;;
        esac
    done

    # Recalculate DP after overrides
    DP_SIZE=$((TOTAL_GPUS / (TP_SIZE * PP_SIZE)))

    # Validate configuration
    validate_parallelism_config "$TP_SIZE" "$PP_SIZE" "$NUM_LAYERS" "$NUM_ATTENTION_HEADS" "$TOTAL_GPUS" "$SEQ_LENGTH"
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Auto-adjust sequence length if needed
    if [ "$TP_SIZE" -gt 1 ] && [ $((SEQ_LENGTH % TP_SIZE)) -ne 0 ]; then
        SEQ_LENGTH=$((SEQ_LENGTH / TP_SIZE * TP_SIZE))
    fi

    # Validate batch size divisibility
    if [ $((GLOBAL_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
        echo "WARNING: Global batch size ($GLOBAL_BATCH_SIZE) is not divisible by DP size ($DP_SIZE)"
        echo "  This may cause issues. Consider adjusting batch size."
    fi

    return 0
}

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print_config() {
    cat <<EOF
╔════════════════════════════════════════════════════════════════════════════╗
║                   TRAINING CONFIGURATION SUMMARY                          ║
╚════════════════════════════════════════════════════════════════════════════╝

PARALLELISM:
  TP (Tensor Parallelism):        $TP_SIZE
  PP (Pipeline Parallelism):      $PP_SIZE
  DP (Data Parallelism):          $DP_SIZE
  Total GPUs:                     $TOTAL_GPUS

BATCH CONFIGURATION:
  Micro Batch Size (per GPU):     $MICRO_BATCH_SIZE
  Global Batch Size:              $GLOBAL_BATCH_SIZE
  Sequence Length:                $SEQ_LENGTH

TRAINING:
  Train Iterations:               $TRAIN_ITERS
  Learning Rate:                  $LEARNING_RATE
  Warmup Iterations:              $LR_WARMUP_ITERS
  LR Decay Style:                 $LR_DECAY_STYLE

PRECISION & OPTIMIZATION:
  FP16:                           $USE_FP16
  Recompute Activations:          $RECOMPUTE_ACTIVATIONS
  Distributed Optimizer:          $USE_DISTRIBUTED_OPTIMIZER

PATHS:
  Singularity Image:              $SINGULARITY_IMAGE
  Megatron Path:                  $MEGATRON_PATH
  Tokenizer Path:                 $TOKENIZER_PATH
  Checkpoint Path:                $CHECKPOINT_PATH

EOF
}

################################################################################
# End of configs.sh
################################################################################
