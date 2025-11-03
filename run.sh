#!/bin/bash

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source configuration
if [ ! -f "$SCRIPT_DIR/configs.sh" ]; then
    echo "ERROR: configs.sh not found in $SCRIPT_DIR"
    exit 1
fi

# Source config with all defaults
source "$SCRIPT_DIR/configs.sh"

# Apply CLI argument overrides
if [ $# -gt 0 ]; then
    if ! apply_config_overrides "$@"; then
        echo "ERROR: Configuration validation failed"
        exit 1
    fi
fi

# Validate Singularity image exists
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity image not found: $SINGULARITY_IMAGE"
    exit 1
fi

# Validate Megatron-LM exists
if [ ! -d "$MEGATRON_PATH" ]; then
    echo "ERROR: Megatron-LM path not found: $MEGATRON_PATH"
    exit 1
fi

# Create output directories
mkdir -p logs

# Print configuration
print_config

# ============================================================================
# CREATE TRAINING WRAPPER SCRIPT
# ============================================================================

TRAIN_WRAPPER_PATH="/tmp/train_wrapper_$$.sh"
cat > "$TRAIN_WRAPPER_PATH" << WRAPPER_EOF
#!/bin/bash
cd /workspace
torchrun --nproc_per_node=8 ./Megatron-LM/pretrain_gpt.py \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --num-query-groups $NUM_ATTENTION_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --vocab-size $VOCAB_SIZE \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./llama2_tokenizer/tokenizer.model \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --lr $LEARNING_RATE \
    --lr-decay-style $LR_DECAY_STYLE \
    --adam-beta1 $ADAM_BETA1 \
    --adam-beta2 $ADAM_BETA2 \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $CLIP_GRAD \
    --fp16 \
    --recompute-activations \
    --mock-data \
    --data-cache-path $DATA_CACHE_PATH \
    --split 99,1,0 \
    --log-interval $LOG_INTERVAL \
    --log-throughput \
    --eval-interval 1000000 \
    --eval-iters 1
WRAPPER_EOF

chmod +x "$TRAIN_WRAPPER_PATH"

# ============================================================================
# EXECUTE TRAINING VIA SINGULARITY
# ============================================================================

START_TIME=$(date +%s)

echo "Starting training at $(date)"
echo ""
echo "Configuration applied. Beginning training..."
echo ""

# Run singularity with training
singularity exec \
    --bind "$(pwd):/workspace" \
    --bind "$TRAIN_WRAPPER_PATH:/train_wrapper.sh" \
    --bind "/usr/lib64/libnvidia-ml.so.1:/usr/lib/libnvidia-ml.so.1" \
    --pwd "/workspace" \
    --env "NUM_LAYERS=$NUM_LAYERS,HIDDEN_SIZE=$HIDDEN_SIZE,NUM_ATTENTION_HEADS=$NUM_ATTENTION_HEADS,FFN_HIDDEN_SIZE=$FFN_HIDDEN_SIZE,SEQ_LENGTH=$SEQ_LENGTH,MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS,VOCAB_SIZE=$VOCAB_SIZE,TP_SIZE=$TP_SIZE,PP_SIZE=$PP_SIZE,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE,TRAIN_ITERS=$TRAIN_ITERS,LR_WARMUP_ITERS=$LR_WARMUP_ITERS,LEARNING_RATE=$LEARNING_RATE,MIN_LR=$MIN_LR,LR_DECAY_STYLE=$LR_DECAY_STYLE,ADAM_BETA1=$ADAM_BETA1,ADAM_BETA2=$ADAM_BETA2,WEIGHT_DECAY=$WEIGHT_DECAY,CLIP_GRAD=$CLIP_GRAD,DATA_CACHE_PATH=$DATA_CACHE_PATH,LOG_INTERVAL=$LOG_INTERVAL,TENSORBOARD_DIR=$TENSORBOARD_DIR,SAVE_INTERVAL=$SAVE_INTERVAL,EVAL_INTERVAL=$EVAL_INTERVAL,EVAL_ITERS=$EVAL_ITERS" \
    "$SINGULARITY_IMAGE" \
    bash /train_wrapper.sh

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# ============================================================================
# POST-TRAINING SUMMARY
# ============================================================================

echo ""
echo "Training done in $ELAPSED_TIME seconds."

# Cleanup temporary wrapper script
rm -f "$TRAIN_WRAPPER_PATH"

exit $TRAIN_EXIT_CODE