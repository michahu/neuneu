"""
Simplified training script using HuggingFace Trainer with custom evaluation for per-token losses.
"""

import argparse
import logging
import os
import re
import tempfile

import numpy as np
import torch
from hf_olmo import OLMoForCausalLM  # registers OLMo with AutoModelForCausalLM
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from src.custom_trainer import CustomTrainer, ShardAwareDataCollator
from src.token_dataset import TextDataset, TokenDataset


def get_checkpoint_dirs(model_directory):
    checkpoints = [
        item
        for item in os.listdir(model_directory)
        if os.path.isdir(os.path.join(model_directory, item))
        and item.startswith("checkpoint-")
    ]

    return sorted(checkpoints, key=lambda x: int(x.split("-")[1]))


def eval(model, tokenizer, args, eval_dataset, results_dir):
    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
    )

    data_collator = ShardAwareDataCollator(tokenizer)

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_save_dir=results_dir if eval_dataset else None,
        max_seq_len=args.seq_len,
        eval_only=True,
        use_shard_batching=True,
        data_collator=data_collator,
        test_mode=args.test_mode,
        tokenizer=tokenizer,
        overwrite_eval_losses=args.overwrite_eval_losses,
    )

    trainer.evaluate()


def _main(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create datasets
    logger.info("Loading datasets...")

    # Check for required files
    val_dir = os.path.join(args.data_dir, "val")

    eval_dataset = None

    # Skip TokenDataset creation if using --hf_model (it will create TextDataset instead)
    if args.hf_model is None and os.path.exists(val_dir):
        # Create PyTorch dataset first
        eval_dataset = TokenDataset(
            data_dir=val_dir,
            max_seq_len=args.seq_len,
            split="val",
            max_shards=args.max_shards,
            shard_number=args.shard_number,
        )

        logger.info(f"Loaded validation dataset with {len(eval_dataset)} documents")
        logger.info(f"Total validation tokens: {eval_dataset.total_tokens} tokens")
    elif args.hf_model is None:
        logger.warning("No validation dataset found")

    # Mode 1: Evaluate a HuggingFace model with different tokenizer (e.g., Pythia)
    if args.hf_model is not None:
        logger.info(f"Evaluating HuggingFace model: {args.hf_model}")
        logger.info(f"Source tokenizer (for stored tokens): {args.source_tokenizer}")

        source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)

        # Load target model and tokenizer
        revision = args.revision if args.revision else None
        logger.info(f"Loading model revision: {revision or 'main'}")

        target_tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model, revision=revision
        )
        if target_tokenizer.pad_token is None:
            target_tokenizer.pad_token = target_tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(args.hf_model, revision=revision)

        # Create TextDataset that decodes and re-tokenizes
        eval_dataset = TextDataset(
            data_dir=val_dir,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_seq_len=args.seq_len,
            split="val",
            max_shards=args.max_shards,
            shard_number=args.shard_number,
        )

        logger.info(f"Loaded validation dataset with {len(eval_dataset)} documents")

        # Create output directory
        # e.g., results/hf_eval/EleutherAI--pythia-70m/step1000/
        model_name_safe = args.hf_model.replace("/", "--")
        revision_safe = (args.revision or "main").replace("/", "--")
        eval_results_dir = os.path.join(
            args.eval_base_results_dir, "hf_eval", model_name_safe, revision_safe
        )
        os.makedirs(eval_results_dir, exist_ok=True)

        logger.info(f"Saving results to {eval_results_dir}")

        eval(model, target_tokenizer, args, eval_dataset, eval_results_dir)
        return

    else:
        logger.info(f"Evaluating DataDecide models")
        assert args.datadecide_pretraining_recipe is not None, (
            "DataDecide pretraining recipe must be provided"
        )

        model_name = (
            f"DataDecide-{args.datadecide_pretraining_recipe}-{args.model_size}"
        )

        # Get all checkpoints for this model
        REPO_ID = "allenai/" + model_name  # <-- change to the exact model repo

        assert args.seed_name != "None", (
            "Seed name must be provided for DataDecide models"
        )
        if args.seed_name == "seed-default":
            SEED_NAME = "seed-default"
        elif (
            args.model_size == "1B"
        ):  # for some reason the seed name is different for the 1B model
            SEED_NAME = f"seed-large-aux-{args.seed_name}"
        elif args.model_size in ["90M", "150M", "300M", "530M", "750M"]:
            SEED_NAME = f"seed-small-aux-{args.seed_name}"
        else:
            raise ValueError(
                f"Something's wrong with either the model size {args.model_size} or the seed name {args.seed_name}"
            )  # e.g. "seed-default", "seed-0", "seed-1", etc.

        # 1) list all refs (tags) and keep only those matching the seed you want
        api = HfApi()
        refs = api.list_repo_refs(REPO_ID)
        # typical tag format: step69369-seed-default  (sometimes seed-0, seed-1)

        pattern = re.compile(r"^step\d+-" + re.escape(SEED_NAME) + r"$")

        step_tags = [t.name for t in refs.branches if pattern.match(t.name)]
        # Sort numerically by the step number
        step_tags = sorted(
            step_tags, key=lambda s: int(re.search(r"step(\d+)", s).group(1))
        )

        logger.info(f"Found {len(step_tags)} checkpoints for {SEED_NAME}")

        if args.checkpoint_dir is not None:
            step_tags = [args.checkpoint_dir]

        if args.trained_model_start_idx is not None:
            assert args.num_models_to_eval is not None, (
                "If trained_model_start_idx is provided, num_models_to_eval must also be provided"
            )
            step_tags = step_tags[
                args.trained_model_start_idx : args.trained_model_start_idx
                + args.num_models_to_eval
            ]

        logger.info(f"Evaluating {len(step_tags)} checkpoints")

        for tag in step_tags:
            # Check for skip BEFORE loading the model
            eval_results_dir = f"{args.eval_base_results_dir}/{model_name}/{tag}"
            if args.shard_number is not None:
                eval_file = os.path.join(
                    eval_results_dir, f"eval_losses_shard_{args.shard_number}.npy"
                )
                if os.path.exists(eval_file):
                    logging.info(
                        f"Skipping {tag} for shard {args.shard_number} (already exists)"
                    )
                    continue

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, tag)
                model = OLMoForCausalLM.from_pretrained(
                    REPO_ID,
                    revision=tag,
                    cache_dir=model_path,
                )

            # Use standard OLMo tokenizer - all DataDecide models use the same tokenizer
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B")

            os.makedirs(eval_results_dir, exist_ok=True)
            logger.info(f"Evaluating {tag} in {eval_results_dir}")
            eval(model, tokenizer, args, eval_dataset, eval_results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train language model with HuggingFace Trainer"
    )

    parser.add_argument(
        "--eval_base_results_dir", help="Base directory to save eval results"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir", default="./data", help="Directory with tokenized data"
    )
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to evaluate",
    )
    parser.add_argument(
        "--shard_number",
        type=str,
        default=None,
        help="Shard number to evaluate. Mostly for debugging.",
    )
    parser.add_argument(
        "--overwrite_eval_losses",
        action="store_true",
        help="Overwrite eval losses. Otherwise, will skip if eval losses already exist.",
    )

    # Model arguments
    parser.add_argument(
        "--model_name", default="distilbert/distilgpt2", help="Model name"
    )
    parser.add_argument(
        "--trained_model_dir", default=None, help="Directory to load trained model from"
    )
    parser.add_argument(
        "--trained_model_start_idx",
        type=int,
        default=None,
        help="Start index of trained model",
    )
    parser.add_argument(
        "--num_models_to_eval",
        type=int,
        default=None,
        help="Number of models to evaluate",
    )
    parser.add_argument("--model_size", type=str, default="4M", help="Model size")
    parser.add_argument("--seed_name", type=str, default="None", help="Seed name")
    parser.add_argument(
        "--datadecide_pretraining_recipe",
        type=str,
        default=None,
        help="DataDecide pretraining recipe",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory to evaluate",
    )

    # HuggingFace model evaluation (for models with different tokenizers)
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HuggingFace model to evaluate (e.g., 'EleutherAI/pythia-70m'). "
        "Will decode stored tokens and re-tokenize with this model's tokenizer.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/checkpoint (e.g., 'step1000', 'main'). Default: main branch.",
    )
    parser.add_argument(
        "--source_tokenizer",
        type=str,
        default="allenai/OLMo-1B",
        help="Tokenizer used for the stored tokens (default: allenai/OLMo-1B)",
    )

    # Training arguments
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=64, help="Train batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=64, help="Eval batch size"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Logging frequency"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Evaluation frequency"
    )
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    parser.add_argument(
        "--learning_rate", type=float, default=0.00025, help="Learning rate"
    )
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")

    # Debugging arguments
    parser.add_argument("--test_mode", action="store_true", help="Test mode")

    args = parser.parse_args()

    _main(args)


if __name__ == "__main__":
    main()
