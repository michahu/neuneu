#!/usr/bin/env python
"""
Batch conversion script: Convert token-level losses to word-level losses.

This script processes all model/step combinations in the results directory,
converting per-token losses to whitespace-word losses for tokenizer-invariant
meta-model training.

Optimization: Since all loss files for a given shard share the same tokens,
we precompute the token-to-word mapping once per shard and reuse it for all
loss files, avoiding redundant tokenization and alignment work.

Usage:
    python scripts/convert_to_word_losses.py \
        --data-dir data/datadecide_subset/val \
        --results-dir results/datadecide_eval \
        --tokenizer allenai/OLMo-1B

Output:
    For each eval_losses_shard_*.npy file, creates:
    - word_losses_shard_*.npy: Converted word-level losses
    - word_losses_shard_*.json: Metadata (tokenizer, stats, boundaries)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta.word_loss_converter import (
    WordLossConverter,
    align_tokens_to_words,
    whitespace_tokenize,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_loss_files(results_dir: Path) -> List[Path]:
    """
    Find all eval_losses_shard_*.npy files in the results directory.

    Returns:
        List of paths to loss files
    """
    loss_files = []
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for step_dir in model_dir.iterdir():
            if not step_dir.is_dir() or not step_dir.name.startswith("step"):
                continue
            for npy_file in step_dir.glob("eval_losses_shard_*.npy"):
                loss_files.append(npy_file)
    return sorted(loss_files)


def get_output_path(loss_file: Path) -> Path:
    """
    Get the output path for a converted loss file.

    eval_losses_shard_00000141.npy -> word_losses_shard_00000141.npy
    """
    filename = loss_file.name.replace("eval_losses", "word_losses")
    return loss_file.parent / filename


def get_shard_number(loss_file: Path) -> str:
    """Extract shard number from filename."""
    # eval_losses_shard_00000141.npy -> 00000141
    name = loss_file.stem  # eval_losses_shard_00000141
    parts = name.split("_")
    return parts[-1] if parts else "unknown"


def precompute_word_mapping(
    tokens: np.ndarray,
    boundaries: np.ndarray,
    converter: WordLossConverter,
    max_docs: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], List[Tuple[int, int]], Dict]:
    """
    Precompute token-to-word mapping for all documents in a shard.

    This is the expensive operation that we only need to do once per shard,
    since all loss files share the same tokens.

    The mapping is stored as (token_indices, word_ids, num_words) where:
    - token_indices: flat array of all token indices that belong to words
    - word_ids: corresponding word index for each token
    - num_words: total number of words in the document

    This allows vectorized aggregation via np.bincount.

    Args:
        tokens: 1D array of token IDs for entire shard
        boundaries: Document boundary indices
        converter: WordLossConverter with loaded tokenizer
        max_docs: Optional limit on documents to process

    Returns:
        Tuple of:
        - doc_mappings: List of (token_indices, word_ids, num_words) per doc
        - doc_ranges: List of (start, end) token indices for each doc
        - metadata: Stats about the precomputation
    """
    doc_mappings = []
    doc_ranges = []

    num_docs = len(boundaries) - 1 if boundaries[-1] > boundaries[-2] else len(boundaries)
    if max_docs is not None:
        num_docs = min(num_docs, max_docs)

    total_tokens = 0
    total_words = 0

    for doc_idx in range(num_docs):
        start = boundaries[doc_idx]
        end = boundaries[doc_idx + 1] if doc_idx + 1 < len(boundaries) else len(tokens)

        doc_tokens = tokens[start:end]
        if len(doc_tokens) == 0:
            continue

        # Decode and get offset mapping (expensive - only do once)
        text = converter.decode_tokens(doc_tokens)
        token_offsets = converter.get_offset_mapping(text)

        # Handle length mismatch
        if len(token_offsets) != len(doc_tokens):
            min_len = min(len(token_offsets), len(doc_tokens))
            token_offsets = token_offsets[:min_len]
            # Adjust end to match
            end = start + min_len

        # Whitespace tokenize
        word_spans = [(ws, we) for _, ws, we in whitespace_tokenize(text)]

        if len(word_spans) == 0:
            continue

        # Align tokens to words (expensive - only do once)
        word_to_tokens = align_tokens_to_words(token_offsets, word_spans)

        # Convert to vectorized format: (token_indices, word_ids, num_words)
        # This allows fast aggregation via np.bincount
        token_indices_list = []
        word_ids_list = []
        for word_idx, tok_indices in enumerate(word_to_tokens):
            for tok_idx in tok_indices:
                token_indices_list.append(tok_idx)
                word_ids_list.append(word_idx)

        token_indices = np.array(token_indices_list, dtype=np.int32)
        word_ids = np.array(word_ids_list, dtype=np.int32)
        num_words = len(word_to_tokens)

        doc_mappings.append((token_indices, word_ids, num_words))
        doc_ranges.append((start, end))
        total_tokens += end - start
        total_words += num_words

    metadata = {
        "num_docs": len(doc_mappings),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "compression_ratio": total_tokens / total_words if total_words > 0 else 0.0,
    }

    return doc_mappings, doc_ranges, metadata


def apply_word_mapping(
    token_losses: np.ndarray,
    doc_mappings: List[Tuple[np.ndarray, np.ndarray, int]],
    doc_ranges: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply precomputed word mapping to convert token losses to word losses.

    Uses vectorized np.bincount for fast aggregation - O(n) where n is number
    of tokens, with no Python loops over words.

    Args:
        token_losses: 1D array of per-token losses
        doc_mappings: List of (token_indices, word_ids, num_words) per document
        doc_ranges: Token index ranges for each document

    Returns:
        Tuple of:
        - concatenated_losses: All word losses concatenated
        - word_boundaries: Start index of each document in concatenated array
        - num_words_per_doc: Number of words in each document
    """
    all_word_losses = []
    word_boundaries = [0]
    num_words_per_doc = []

    for (token_indices, word_ids, num_words), (start, end) in zip(doc_mappings, doc_ranges):
        # Check if document fits in loss array
        if end > len(token_losses):
            break

        doc_losses = token_losses[start:end]
        doc_losses = np.nan_to_num(doc_losses, nan=0.0, posinf=10.0, neginf=0.0)

        # Vectorized aggregation: sum token losses by word using bincount
        # token_indices indexes into doc_losses, word_ids tells which word each belongs to
        if len(token_indices) > 0:
            token_loss_values = doc_losses[token_indices]
            word_losses = np.bincount(word_ids, weights=token_loss_values, minlength=num_words)
        else:
            word_losses = np.zeros(num_words, dtype=np.float64)

        all_word_losses.append(word_losses.astype(np.float32))
        word_boundaries.append(word_boundaries[-1] + num_words)
        num_words_per_doc.append(num_words)

    if all_word_losses:
        concatenated_losses = np.concatenate(all_word_losses)
    else:
        concatenated_losses = np.array([], dtype=np.float32)

    return (
        concatenated_losses,
        np.array(word_boundaries, dtype=np.int64),
        np.array(num_words_per_doc, dtype=np.int32),
    )


def convert_all(
    data_dir: Path,
    results_dir: Path,
    tokenizer_name: str,
    overwrite: bool = False,
    max_docs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Convert all loss files in the results directory.

    Optimized: Groups files by shard and precomputes the token-to-word mapping
    once per shard, then applies it to all loss files for that shard.

    Args:
        data_dir: Directory containing tokens and boundaries files
        results_dir: Directory containing model evaluation results
        tokenizer_name: HuggingFace tokenizer name
        overwrite: Whether to overwrite existing converted files
        max_docs: Optional limit on documents per file (for testing)
        dry_run: If True, just list files without converting

    Returns:
        Summary statistics
    """
    # Find all loss files
    loss_files = find_loss_files(results_dir)
    logger.info(f"Found {len(loss_files)} loss files to convert")

    if not loss_files:
        logger.warning("No loss files found. Check your results directory.")
        return {"converted": 0, "skipped": 0, "failed": 0}

    # Find tokens and boundaries files
    tokens_files = list(data_dir.glob("*_tokens.npy"))
    boundaries_files = list(data_dir.glob("*_boundaries.npy"))

    if not tokens_files:
        logger.error(f"No tokens files found in {data_dir}")
        return {"converted": 0, "skipped": 0, "failed": 0}

    if not boundaries_files:
        logger.error(f"No boundaries files found in {data_dir}")
        return {"converted": 0, "skipped": 0, "failed": 0}

    # Map shard numbers to data files
    shard_to_data = {}
    for tokens_file in tokens_files:
        parts = tokens_file.stem.split("_")
        if len(parts) >= 2:
            shard_num = parts[-2]
            boundaries_file = tokens_file.parent / tokens_file.name.replace("_tokens.npy", "_boundaries.npy")
            if boundaries_file.exists():
                shard_to_data[shard_num] = {
                    "tokens": tokens_file,
                    "boundaries": boundaries_file,
                }

    logger.info(f"Found {len(shard_to_data)} shard data files")

    # Group loss files by shard for efficient batch processing
    shard_to_loss_files: Dict[str, List[Path]] = {}
    for loss_file in loss_files:
        shard_num = get_shard_number(loss_file)
        if shard_num not in shard_to_loss_files:
            shard_to_loss_files[shard_num] = []
        shard_to_loss_files[shard_num].append(loss_file)

    if dry_run:
        logger.info("Dry run - listing files only:")
        for loss_file in loss_files[:10]:
            output_path = get_output_path(loss_file)
            logger.info(f"  {loss_file} -> {output_path}")
        if len(loss_files) > 10:
            logger.info(f"  ... and {len(loss_files) - 10} more")
        return {"converted": 0, "skipped": 0, "failed": 0, "total": len(loss_files)}

    # Initialize converter once
    converter = WordLossConverter(tokenizer_name)

    stats = {"converted": 0, "skipped": 0, "failed": 0}
    total = len(loss_files)
    processed = 0

    # Process each shard
    for shard_num, shard_loss_files in shard_to_loss_files.items():
        if shard_num not in shard_to_data:
            logger.warning(f"No data files found for shard {shard_num}, skipping {len(shard_loss_files)} files")
            stats["failed"] += len(shard_loss_files)
            processed += len(shard_loss_files)
            continue

        data_files = shard_to_data[shard_num]

        # Filter to files that need processing
        files_to_process = []
        for loss_file in shard_loss_files:
            output_path = get_output_path(loss_file)
            if output_path.exists() and not overwrite:
                stats["skipped"] += 1
            else:
                files_to_process.append(loss_file)

        if not files_to_process:
            processed += len(shard_loss_files)
            continue

        # Load shard data and precompute mapping ONCE
        logger.info(f"Precomputing word mapping for shard {shard_num} ({len(files_to_process)} files to convert)")
        tokens = np.load(data_files["tokens"])
        boundaries = np.load(data_files["boundaries"])

        try:
            doc_word_to_tokens, doc_ranges, precompute_meta = precompute_word_mapping(
                tokens, boundaries, converter, max_docs
            )
            logger.info(
                f"  Shard {shard_num}: {precompute_meta['num_docs']} docs, "
                f"{precompute_meta['total_tokens']} tokens -> {precompute_meta['total_words']} words"
            )
        except Exception as e:
            logger.error(f"Failed to precompute mapping for shard {shard_num}: {e}")
            stats["failed"] += len(files_to_process)
            processed += len(shard_loss_files)
            continue

        # Apply mapping to each loss file (fast operation)
        for loss_file in files_to_process:
            output_path = get_output_path(loss_file)
            processed += 1

            if processed % 50 == 0 or processed == 1:
                logger.info(f"Processing {processed}/{total}: {loss_file.parent.parent.name}/{loss_file.parent.name}")

            try:
                token_losses = np.load(loss_file)

                concatenated_losses, word_boundaries, num_words_per_doc = apply_word_mapping(
                    token_losses, doc_word_to_tokens, doc_ranges
                )

                # Save output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, concatenated_losses)

                # Save metadata
                metadata_file = output_path.with_suffix('.json')
                metadata = {
                    "tokenizer_name": tokenizer_name,
                    "total_tokens": int(precompute_meta["total_tokens"]),
                    "total_words": int(len(concatenated_losses)),
                    "compression_ratio": precompute_meta["compression_ratio"],
                    "num_docs": len(num_words_per_doc),
                    "skipped_docs": precompute_meta["num_docs"] - len(num_words_per_doc),
                    "word_boundaries": word_boundaries.tolist(),
                    "num_words_per_doc": num_words_per_doc.tolist(),
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                stats["converted"] += 1
            except Exception as e:
                logger.error(f"Failed to convert {loss_file}: {e}")
                stats["failed"] += 1

        # Update processed count for skipped files
        processed += len(shard_loss_files) - len(files_to_process)

    # Final summary
    logger.info(f"\nConversion complete:")
    logger.info(f"  Converted: {stats['converted']}")
    logger.info(f"  Skipped (already exist): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert token-level losses to word-level losses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all files
    python scripts/convert_to_word_losses.py \\
        --data-dir data/datadecide_subset/val \\
        --results-dir results/datadecide_eval

    # Test on first 10 documents per file
    python scripts/convert_to_word_losses.py \\
        --data-dir data/datadecide_subset/val \\
        --results-dir results/datadecide_eval \\
        --max-docs 10

    # Dry run to see what would be converted
    python scripts/convert_to_word_losses.py \\
        --data-dir data/datadecide_subset/val \\
        --results-dir results/datadecide_eval \\
        --dry-run
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing tokens and boundaries .npy files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing model evaluation results"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-1B",
        help="HuggingFace tokenizer name (default: allenai/OLMo-1B)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to convert per file (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without converting"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate directories
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Run conversion
    stats = convert_all(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        tokenizer_name=args.tokenizer,
        overwrite=args.overwrite,
        max_docs=args.max_docs,
        dry_run=args.dry_run,
    )

    # Exit with error code if any failures
    if stats.get("failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
