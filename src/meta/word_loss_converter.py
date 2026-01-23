"""
Convert token-level losses to whitespace-word losses.

This module provides functionality to convert per-token cross-entropy losses
to a tokenizer-invariant representation based on whitespace-separated words.
This enables zero-shot transfer of the meta-model across different tokenizers.

Key insight: By aggregating subword token losses to word-level losses using
sum (equivalent to product of probabilities in log space), we create a
representation that is independent of how each tokenizer splits words.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def whitespace_tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text on whitespace, returning word spans.

    Uses regex pattern ``\\S+`` to find all non-whitespace sequences.
    This preserves contractions (e.g., "don't" stays as one word) and
    keeps punctuation attached to words.

    Args:
        text: Input text to tokenize

    Returns:
        List of (word, start_char, end_char) tuples
    """
    words = []
    for match in re.finditer(r'\S+', text):
        words.append((match.group(), match.start(), match.end()))
    return words


def align_tokens_to_words(
    token_offsets: List[Tuple[int, int]],
    word_spans: List[Tuple[int, int]],
) -> List[List[int]]:
    """
    Map each whitespace word to the token indices that overlap with it.

    A token is assigned to a word if their character spans overlap.
    Tokens with (0, 0) offset (typically special tokens like BOS/EOS) are skipped.

    Args:
        token_offsets: List of (start, end) character offsets for each token
        word_spans: List of (start, end) character offsets for each word

    Returns:
        List where word_to_tokens[i] contains the token indices for word i
    """
    word_to_tokens: List[List[int]] = [[] for _ in word_spans]

    for tok_idx, (tok_start, tok_end) in enumerate(token_offsets):
        # Skip special tokens with (0, 0) offset
        if tok_start == tok_end:
            continue

        for word_idx, (word_start, word_end) in enumerate(word_spans):
            # Check for overlap: token and word spans intersect
            if tok_start < word_end and tok_end > word_start:
                word_to_tokens[word_idx].append(tok_idx)

    return word_to_tokens


def aggregate_losses(
    token_losses: np.ndarray,
    word_to_tokens: List[List[int]],
) -> np.ndarray:
    """
    Aggregate token losses to word losses using SUM.

    Sum of log-losses is equivalent to product of probabilities in log space:
    -log(P(word)) = -log(P(t1) * P(t2) * ...) = -log(P(t1)) + -log(P(t2)) + ...

    Args:
        token_losses: 1D array of per-token cross-entropy losses
        word_to_tokens: Mapping from word index to list of token indices

    Returns:
        1D array of per-word losses (summed from constituent tokens)
    """
    word_losses = []
    for token_indices in word_to_tokens:
        if len(token_indices) > 0:
            # SUM of log-losses = joint probability in log space
            word_loss = np.sum(token_losses[token_indices])
        else:
            # No tokens for this word (edge case - shouldn't happen normally)
            word_loss = 0.0
        word_losses.append(word_loss)

    return np.array(word_losses, dtype=np.float32)


class WordLossConverter:
    """
    Convert token-level losses to whitespace-word losses.

    This class handles the full conversion pipeline:
    1. Decode tokens to text using the tokenizer
    2. Re-tokenize to get character offsets for each token
    3. Split text on whitespace to get word boundaries
    4. Align tokens to words based on character overlap
    5. Aggregate token losses to word losses using sum

    Example:
        converter = WordLossConverter("allenai/OLMo-1B")
        word_losses = converter.convert_document(token_ids, token_losses)
    """

    def __init__(self, tokenizer_name: str = "allenai/OLMo-1B"):
        """
        Initialize the converter with a tokenizer.

        Args:
            tokenizer_name: HuggingFace tokenizer name or path
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self._load_tokenizer(tokenizer_name)

    def _load_tokenizer(self, tokenizer_name: str):
        """Load tokenizer, handling OLMo specially.

        OLMo uses the same tokenizer as GPT-NeoX, so we can use GPTNeoXTokenizerFast
        directly to avoid the hf_olmo dependency and trust_remote_code prompts.
        """
        if "olmo" in tokenizer_name.lower():
            from transformers import GPTNeoXTokenizerFast
            # OLMo uses the same tokenizer as GPT-NeoX (EleutherAI/gpt-neox-20b)
            return GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name)

    def decode_tokens(self, token_ids: np.ndarray) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: 1D array of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)

    def get_offset_mapping(self, text: str) -> List[Tuple[int, int]]:
        """
        Re-tokenize text and get character offsets for each token.

        Args:
            text: Input text

        Returns:
            List of (start, end) character offsets for each token
        """
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return encoded['offset_mapping']

    def convert_document(
        self,
        token_ids: np.ndarray,
        token_losses: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert a single document from token losses to word losses.

        Args:
            token_ids: 1D array of token IDs for the document
            token_losses: 1D array of per-token losses (same length as token_ids)

        Returns:
            Tuple of:
            - word_losses: 1D array of per-word losses
            - metadata: Dict with conversion stats (num_tokens, num_words, etc.)
        """
        if len(token_ids) != len(token_losses):
            raise ValueError(
                f"token_ids length ({len(token_ids)}) != token_losses length ({len(token_losses)})"
            )

        if len(token_ids) == 0:
            return np.array([], dtype=np.float32), {
                "num_tokens": 0,
                "num_words": 0,
                "compression_ratio": 0.0,
            }

        # Clean losses (handle NaN/inf)
        token_losses = np.nan_to_num(token_losses, nan=0.0, posinf=10.0, neginf=0.0)

        # Step 1: Decode tokens to text
        text = self.decode_tokens(token_ids)

        # Step 2: Re-tokenize with offset mapping
        token_offsets = self.get_offset_mapping(text)

        # Verify we get same number of tokens back
        if len(token_offsets) != len(token_ids):
            logger.warning(
                f"Token count mismatch after re-encoding: "
                f"original={len(token_ids)}, re-encoded={len(token_offsets)}. "
                f"Using min length."
            )
            min_len = min(len(token_offsets), len(token_ids))
            token_offsets = token_offsets[:min_len]
            token_losses = token_losses[:min_len]

        # Step 3: Whitespace tokenize
        word_spans = [(start, end) for _, start, end in whitespace_tokenize(text)]

        if len(word_spans) == 0:
            return np.array([], dtype=np.float32), {
                "num_tokens": len(token_ids),
                "num_words": 0,
                "compression_ratio": 0.0,
            }

        # Step 4: Align tokens to words
        word_to_tokens = align_tokens_to_words(token_offsets, word_spans)

        # Step 5: Aggregate losses
        word_losses = aggregate_losses(token_losses, word_to_tokens)

        # Compute metadata
        metadata = {
            "num_tokens": len(token_ids),
            "num_words": len(word_losses),
            "compression_ratio": len(token_ids) / len(word_losses) if len(word_losses) > 0 else 0.0,
            "token_loss_sum": float(np.sum(token_losses)),
            "word_loss_sum": float(np.sum(word_losses)),
        }

        return word_losses, metadata


def convert_shard(
    tokens_file: str,
    boundaries_file: str,
    loss_file: str,
    output_file: str,
    tokenizer_name: str = "allenai/OLMo-1B",
    max_docs: Optional[int] = None,
) -> Dict:
    """
    Convert all documents in a shard from token losses to word losses.

    Args:
        tokens_file: Path to tokens .npy file (1D array of token IDs)
        boundaries_file: Path to boundaries .npy file (document start indices)
        loss_file: Path to token losses .npy file
        output_file: Path to output .npy file for word losses
        tokenizer_name: HuggingFace tokenizer name
        max_docs: Optional limit on number of documents to convert

    Returns:
        Dict with conversion statistics
    """
    # Load data
    tokens = np.load(tokens_file)
    boundaries = np.load(boundaries_file)
    token_losses = np.load(loss_file)

    # Handle flat token losses (may be longer/shorter than tokens)
    if len(token_losses) < len(tokens):
        logger.warning(
            f"Token losses ({len(token_losses)}) shorter than tokens ({len(tokens)}). "
            f"Only converting documents that fit within losses."
        )

    # Initialize converter
    converter = WordLossConverter(tokenizer_name)

    # Process each document
    all_word_losses = []
    word_boundaries = [0]
    num_words_per_doc = []

    total_tokens = 0
    total_words = 0
    skipped_docs = 0

    num_docs = len(boundaries) - 1 if boundaries[-1] > boundaries[-2] else len(boundaries)
    if max_docs is not None:
        num_docs = min(num_docs, max_docs)

    for doc_idx in range(num_docs):
        # Get document boundaries
        start = boundaries[doc_idx]
        end = boundaries[doc_idx + 1] if doc_idx + 1 < len(boundaries) else len(tokens)

        # Check if this document fits within the loss array
        if end > len(token_losses):
            logger.debug(f"Document {doc_idx} extends beyond loss array, stopping")
            break

        # Get document data
        doc_tokens = tokens[start:end]
        doc_losses = token_losses[start:end]

        if len(doc_tokens) == 0:
            skipped_docs += 1
            continue

        # Convert document
        word_losses, metadata = converter.convert_document(doc_tokens, doc_losses)

        if len(word_losses) == 0:
            skipped_docs += 1
            continue

        # Accumulate
        all_word_losses.append(word_losses)
        word_boundaries.append(word_boundaries[-1] + len(word_losses))
        num_words_per_doc.append(len(word_losses))

        total_tokens += metadata["num_tokens"]
        total_words += metadata["num_words"]

    # Concatenate all word losses
    if all_word_losses:
        concatenated_losses = np.concatenate(all_word_losses)
    else:
        concatenated_losses = np.array([], dtype=np.float32)

    word_boundaries = np.array(word_boundaries, dtype=np.int64)
    num_words_per_doc = np.array(num_words_per_doc, dtype=np.int32)

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as simple .npy (just the concatenated word losses)
    # This matches the format expected by the existing dataset
    np.save(output_file, concatenated_losses)

    # Also save metadata in a companion file
    metadata_file = output_path.with_suffix('.json')
    metadata = {
        "tokenizer_name": tokenizer_name,
        "total_tokens": int(total_tokens),
        "total_words": int(total_words),
        "compression_ratio": total_tokens / total_words if total_words > 0 else 0.0,
        "num_docs": len(num_words_per_doc),
        "skipped_docs": skipped_docs,
        "word_boundaries": word_boundaries.tolist(),
        "num_words_per_doc": num_words_per_doc.tolist(),
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Converted {len(num_words_per_doc)} docs: "
        f"{total_tokens} tokens -> {total_words} words "
        f"(ratio: {metadata['compression_ratio']:.2f})"
    )

    return metadata
