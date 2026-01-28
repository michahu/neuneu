# Neural Neural Scaling Laws (NeuNeu)

Predicting language model downstream task performance via learning curve extrapolation.
[arXiv](https://arxiv.org/abs/2601.19831)

## Architecture

```
Token Losses (256K) → CNN → Soft Prompt
                                  ↓
Accuracy Context → MLP → [BOS; prompt; context] → Transformer Encoder (BERT-style) → Quantile Predictions
```

- **CNN Encoder**: 4 layers, channels (8→16→32→64), kernel 64, stride 16, AdaptiveMaxPool1D
- **Transformer**: 6 layers, 512 hidden dim, 8 heads, RoPE (rotary position encodings)
- **Output**: Quantile regression (5 quantiles: 0.1, 0.25, 0.5, 0.75, 0.9)

## Installation

### Environment Setup

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Downloading the Trained Model

The trained model (`model/best_model.pt`) is stored with Git LFS. To download it:

```bash
# Install Git LFS if you haven't already
brew install git-lfs  # macOS
# apt install git-lfs  # Ubuntu/Debian

# Initialize LFS and pull the model
git lfs install
git lfs pull
```

If you cloned the repo before installing Git LFS, run `git lfs pull` to fetch the model file.

### Evaluating our Trained Model

To evaluate a trained meta-model on evaluation data, use `src.analysis.eval_scaling_predictions`. The evaluation script expects data organized as `data_dir/model_name/...`.

```bash
# Evaluate the neural meta-model
python -m src.analysis.eval_scaling_predictions neural \
    --model_name DataDecide-c4-300M \
    --checkpoint ./model/best_model.pt \
    --data_dir ./data/ \
    --output_dir ./results/eval \
    --context_ratio 0.2 \
    --prediction_mode anchored
```

The `neural` command evaluates models with CNN/soft-prompt encoders. Other evaluation modes:
- `baseline` - Transformer encoder only (no CNN)
- `probe` - CNN/histogram probe models
- `logistic` - Zero-shot logistic baseline

Options:
- `--context_ratio 0.2` - Fraction of checkpoints used as context
- `--prediction_mode sequential` - Use `sequential` (sliding window) or `anchored` (fixed context)

### Visualizing Results

After evaluation, visualize results with `src.analysis.visualize_scaling_predictions`:

```bash
# Run all visualizations (compare, trajectory, aggregate, errors)
python -m src.analysis.visualize_scaling_predictions all \
    --eval_dirs '{"Neural": "./results/eval"}' \
    --output_dir ./results/viz
```

Visualization commands:
- `all` - Run all visualization commands
- `compare` - Bar chart comparing MAE across conditions/tasks
- `trajectory` - Plot predicted vs actual accuracy curves
- `aggregate` - MAE aggregated across models
- `errors` - Average error per training step

To reproduce all our results, follow the next two sections.

## Training Data Preparation

We use the [DataDecide](https://github.com/allenai/DataDecide) model suite:
- 6 model sizes: 90M, 150M, 300M, 530M, 750M, 1B parameters
- Accuracy evaluations on 66 downstream tasks
  
We compute per-token validation cross-entropy losses on a [WebOrganizer](https://arxiv.org/abs/2502.10341) validation shard. If you wish to use the same dataset, do the following:

### 1. Tokenization

```bash
python scripts/tokenize_and_save_corpus.py \
    --output_dir /path/to/tokenized/data \
    --tokenizer_name datadecide \
    --split train \
    --weborganizer
```
### 2. Model Evaluation

Run this to evaluate all DataDecide models of a particular seed on a given validation corpus. We used shard 00000141 (randomly chosen) from WebOrganizer, which we've included in `data/train/` for your convenience.

```bash
# Submit array job for all model sizes
sbatch scripts/eval_datadecide.slurm
```

## Meta-Model Training

The main experiment trains a neural network to predict future task accuracy from:
1. **Temporal context**: Sequence of observed accuracies at past checkpoints
2. **Token-level losses**: Per-token cross-entropy losses from a validation set

```bash
# Train the meta-model
sbatch scripts/train_metaloss.slurm

# Or run directly
python -m src.meta.train --help
```

See `scripts/train/full.sh` to reproduce our results.


## Scaling Law Evaluation


Run evaluation on a cluster to compute predictions across all model sizes and seeds:

```bash
bash scripts/eval/full.sh
```


## Directory Structure

```
simple-perplexity/
├── src/
│   ├── meta/                    # Meta-predictor (main experiment)
│   │   ├── datasets/            # Dataset classes
│   │   │   ├── base.py          
│   │   │   ├── meta.py          
│   │   │   └── probe.py         
│   │   ├── model.py             # NeuNeu architecture here
│   │   ├── train.py             # Training loop
│   │   ├── train_probe.py       # Probe training
│   │   ├── probes.py            # Probe models
│   │   ├── utils.py             
│   │   └── word_loss_converter.py
│   ├── analysis/                # Evaluation and visualization
│   │   ├── eval_output.py       
│   │   ├── eval_scaling_predictions.py  # Scaling law evaluation
│   │   ├── visualize_scaling_predictions.py  # Visualization
│   │   └── utils.py             
│   ├── eval.py                  # Data preprocessing
│   ├── custom_trainer.py        
│   ├── token_dataset.py         
│   └── tokenize_and_save_corpus.py 
└── scripts/
    ├── train/                   
    │   ├── full.sh              
    │   ├── baseline.sh          
    │   └── probes.sh            
    ├── eval/                    
    │   ├── full.sh              
    │   ├── baseline.sh          
    │   ├── logistic.sh          
    │   └── probes.sh            
    └── ...                      
```



