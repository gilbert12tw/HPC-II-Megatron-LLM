# Megatron-LM Llama2 7B Pretraining

A framework for experimenting with tensor parallelism (TP), pipeline parallelism (PP), and data parallelism (DP) on 8× V100 GPUs using Megatron-LM.

**For detailed experiment specification and requirements, see [SPEC.md](https://hackmd.io/@LittlePants/SJMGmQI1Zl)**

## Setup

### Prerequisites

- Singularity container: `/work/u4876763/megatron-nv.sif`
- Model Path: `/work/jonathan0hsu/llm-inference/model`
- 8× V100 GPUs available
- Sufficient disk space for data cache

### Directory Structure

```
pretrain_megatron/
├── run.sh                # Main training launcher
├── configs.sh            # Configuration defaults
├── README.md             # This file
├── Megatron-LM/          # Megatron-LM framework
├── llama2_config/        # Model configuration
├── llama2_tokenizer/     # Tokenizer files
├── data_cache/           # Cached preprocessed data
└── logs/                 # Training logs
```

## How to Run

### Basic Training (Default: TP=2, PP=2, DP=2)

```bash
./run.sh
```

### Custom Configuration

Override default settings via CLI arguments:

```bash
# TP=4, PP=1, DP=2
./run.sh --tp 4 --pp 1

# TP=1, PP=4, DP=2
./run.sh --tp 1 --pp 4

# Custom batch sizes
./run.sh --tp 2 --pp 2 --micro-batch 2 --global-batch 8

# Different sequence length and iterations
./run.sh --tp 8 --pp 1 --seq-len 1024 --train-iters 500
```

### Available CLI Options

- `--tp N`: Tensor parallelism size
- `--pp N`: Pipeline parallelism size
- `--micro-batch N`: Micro batch size per GPU
- `--global-batch N`: Global batch size
- `--seq-len N`: Sequence length
- `--train-iters N`: Number of training iterations
- `--lr FLOAT`: Learning rate

### Output

Training logs include:

```
Training done in 72.54 seconds.
Avg Global Tokens/s: 1941.05
Peak GPU Mem (GB): 24.69
```

**Key Metrics:**
- **Global Tokens/s**: Throughput in tokens per second
- **Peak GPU Mem (GB)**: Maximum memory used during training

### Notes

- Create your own job script wrapping `run.sh` and use sbatch to submit
- The system automatically validates parallelism configurations
- Mock data (NLTK synthetic) is enabled by default