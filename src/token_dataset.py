
import argparse
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer,
)
# from trl import SFTTrainer, SFTConfig
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
import os
import logging
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR



class TokenDataset(TorchDataset):
    """Dataset that reads documents from numpy arrays and yields individual documents."""

    def __init__(self, data_dir, max_seq_len=2048, seed=42, split="train", test_mode=False, max_shards=None, shard_number=None):
        
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
    
        self.seed = seed
        self.epoch = 0
        self.test_mode = test_mode
        self.total_tokens = 0
        self.max_shards = max_shards
        self.shard_number = shard_number

        # Find all shard files
        self.shard_files = []
        logging.warning(f"Loading {data_dir}")

        if test_mode:  
            logging.warning(f"Loading {data_dir} -- only loading first 3 shards")
            shard_files = os.listdir(data_dir)[:3]
        else:
            shard_files = os.listdir(data_dir)


        logging.warning(f"Found {len(shard_files)/2} shards total")

        for file in shard_files:

            # if file.endswith("_tokens.npy"):

            if file.endswith("_tokens.npy") and (self.shard_number is None or self.shard_number in file):
        
                shard_number = file.split("_")[1]
                tokens_file = os.path.join(data_dir, file)
                boundaries_file = os.path.join(data_dir, file.replace("_tokens.npy", "_boundaries.npy"))
                # topics_file = os.path.join(
                #     "/Corpus-200B/domains_topics", 
                #     f"CC_shard_{shard_number}_processed__choice.npy",
                # )
                # format_file = os.path.join(
                #     "/Corpus-200B/domains_formats", 
                #     f"CC_shard_{shard_number}_processed__choice.npy",
                # )
                # clusters_file = os.path.join(
                #     "/Corpus-200B/domains_clusters-k24", 
                #     f"CC_shard_{shard_number}_processed.npy",
                # )
                
                if os.path.exists(boundaries_file): 
                # and os.path.exists(topics_file) \
                # and os.path.exists(format_file) \
                # and os.path.exists(clusters_file):
                   
                    self.shard_files.append({
                        'tokens_file': tokens_file,
                        'boundaries_file': boundaries_file,
                        # 'topics_file': topics_file,
                        # 'format_file': format_file,
                        # 'clusters_file': clusters_file,
                        'shard_number': shard_number
                    })
                    

        if self.max_shards is not None: 
            self.shard_files = self.shard_files[:self.max_shards]
    
        # Sort by shard number for consistent ordering
        self.shard_files.sort(key=lambda x: int(x['shard_number']))

        logging.warning(f"Loaded {len(self.shard_files)} shards")
        
        
        # Calculate total documents across all shards
        self.total_docs = 0
        self.shard_info = []
        for shard in self.shard_files:
            boundaries = np.load(shard['boundaries_file'], mmap_mode='r')
            shard_docs = len(boundaries)
            self.shard_info.append({
                'shard': shard,
                'start_doc_idx': self.total_docs,
                'end_doc_idx': self.total_docs + shard_docs,
                'boundaries': boundaries
            })
            self.total_docs += shard_docs
            self.total_tokens += np.sum(boundaries[1:] - boundaries[:-1])

        # Only shuffle if train mode -- don't want to shuffle eval/test data
        if split == "train" or split == "cooldown":    
            self._shuffle_docs()
        else:
            self.shuffled_indices = np.arange(self.total_docs)

    def _shuffle_docs(self):
        """Shuffle document order for current epoch."""
        rng = np.random.RandomState(self.seed + self.epoch)
        self.shuffled_indices = rng.permutation(self.total_docs)

    def set_epoch(self, epoch):
        """Set epoch for shuffling."""
        self.epoch = epoch
        self._shuffle_docs()

    def __len__(self):
        return self.total_docs

    def __getitem__(self, idx):
        # Get shuffled document index
        doc_idx = self.shuffled_indices[idx]
        
        # Find which shard this document belongs to
        shard_idx = None
        for i, info in enumerate(self.shard_info):
            if info['start_doc_idx'] <= doc_idx < info['end_doc_idx']:
                shard_idx = i
                local_doc_idx = doc_idx - info['start_doc_idx']
                break
        
        if shard_idx is None:
            raise IndexError(f"Document index {doc_idx} out of range")
        
        # Load shard data on-demand
        shard_info = self.shard_info[shard_idx]
        tokens = np.load(shard_info['shard']['tokens_file'], mmap_mode='r')
        boundaries = shard_info['boundaries']
        
        # Get document boundaries
        start = boundaries[local_doc_idx]
        end = boundaries[local_doc_idx + 1] if local_doc_idx + 1 < len(boundaries) else len(tokens)
        
        # Truncate if necessary
        end = min(end, start + self.max_seq_len)

        # Extract document tokens
        doc_tokens = tokens[start:end].astype(np.int64)
        num_nonpadding_tokens = len(doc_tokens)

        # Pad if necessary
        if end < start + self.max_seq_len:
            doc_tokens = np.pad(doc_tokens, (0, self.max_seq_len - len(doc_tokens)), mode='constant', constant_values=0)

        labels = doc_tokens.copy()
    
        # Find actual document length (before padding)
        actual_length = len(doc_tokens) if end == start + len(doc_tokens) else end - start

        
        # Set padding positions to -100
        if actual_length < self.max_seq_len:
            labels[actual_length:] = -100

        return {
            'input_ids': doc_tokens.tolist(),
            'labels': labels.tolist(),  # Add this line
            'text': '',
            'shard_number': shard_info['shard']['shard_number'],
            'num_nonpadding_tokens': num_nonpadding_tokens

        }


class TextDataset(TorchDataset):
    """
    Dataset that decodes pre-tokenized documents and re-tokenizes with a different tokenizer.

    This enables evaluating models with different tokenizers (e.g., Pythia) on the same
    text data that was originally tokenized with OLMo's tokenizer.

    The key insight is that we decode the stored tokens back to text, then re-tokenize
    with the target model's tokenizer. This allows fair comparison across models with
    different vocabularies.
    """

    def __init__(
        self,
        data_dir: str,
        source_tokenizer,  # Tokenizer used to create the stored tokens (e.g., OLMo)
        target_tokenizer,  # Tokenizer for the model we're evaluating (e.g., Pythia)
        max_seq_len: int = 2048,
        seed: int = 42,
        split: str = "train",
        test_mode: bool = False,
        max_shards: int = None,
        shard_number: str = None,
    ):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seed = seed
        self.epoch = 0
        self.test_mode = test_mode
        self.total_tokens = 0
        self.max_shards = max_shards
        self.shard_number = shard_number

        # Find all shard files
        self.shard_files = []
        logging.warning(f"Loading {data_dir} (TextDataset with re-tokenization)")

        if test_mode:
            logging.warning(f"Loading {data_dir} -- only loading first 3 shards")
            shard_files = os.listdir(data_dir)[:3]
        else:
            shard_files = os.listdir(data_dir)

        logging.warning(f"Found {len(shard_files)/2} shards total")

        for file in shard_files:
            if file.endswith("_tokens.npy") and (self.shard_number is None or self.shard_number in file):
                shard_number_str = file.split("_")[1]
                tokens_file = os.path.join(data_dir, file)
                boundaries_file = os.path.join(data_dir, file.replace("_tokens.npy", "_boundaries.npy"))

                if os.path.exists(boundaries_file):
                    self.shard_files.append({
                        'tokens_file': tokens_file,
                        'boundaries_file': boundaries_file,
                        'shard_number': shard_number_str
                    })

        if self.max_shards is not None:
            self.shard_files = self.shard_files[:self.max_shards]

        # Sort by shard number for consistent ordering
        self.shard_files.sort(key=lambda x: int(x['shard_number']))

        logging.warning(f"Loaded {len(self.shard_files)} shards")

        # Calculate total documents across all shards
        self.total_docs = 0
        self.shard_info = []
        for shard in self.shard_files:
            boundaries = np.load(shard['boundaries_file'], mmap_mode='r')
            shard_docs = len(boundaries)
            self.shard_info.append({
                'shard': shard,
                'start_doc_idx': self.total_docs,
                'end_doc_idx': self.total_docs + shard_docs,
                'boundaries': boundaries
            })
            self.total_docs += shard_docs
            self.total_tokens += np.sum(boundaries[1:] - boundaries[:-1])

        # Only shuffle if train mode
        if split == "train" or split == "cooldown":
            self._shuffle_docs()
        else:
            self.shuffled_indices = np.arange(self.total_docs)

    def _shuffle_docs(self):
        """Shuffle document order for current epoch."""
        rng = np.random.RandomState(self.seed + self.epoch)
        self.shuffled_indices = rng.permutation(self.total_docs)

    def set_epoch(self, epoch):
        """Set epoch for shuffling."""
        self.epoch = epoch
        self._shuffle_docs()

    def __len__(self):
        return self.total_docs

    def __getitem__(self, idx):
        # Get shuffled document index
        doc_idx = self.shuffled_indices[idx]

        # Find which shard this document belongs to
        shard_idx = None
        for i, info in enumerate(self.shard_info):
            if info['start_doc_idx'] <= doc_idx < info['end_doc_idx']:
                shard_idx = i
                local_doc_idx = doc_idx - info['start_doc_idx']
                break

        if shard_idx is None:
            raise IndexError(f"Document index {doc_idx} out of range")

        # Load shard data on-demand
        shard_info = self.shard_info[shard_idx]
        tokens = np.load(shard_info['shard']['tokens_file'], mmap_mode='r')
        boundaries = shard_info['boundaries']

        # Get document boundaries
        start = boundaries[local_doc_idx]
        end = boundaries[local_doc_idx + 1] if local_doc_idx + 1 < len(boundaries) else len(tokens)

        # Extract source tokens (from stored file)
        source_tokens = tokens[start:end].astype(np.int64)

        # Decode source tokens back to text
        text = self.source_tokenizer.decode(source_tokens.tolist(), skip_special_tokens=False)

        # Re-tokenize with target tokenizer
        encoded = self.target_tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
            add_special_tokens=False,  # Don't add BOS/EOS since text may have them
        )
        doc_tokens = np.array(encoded['input_ids'], dtype=np.int64)
        num_nonpadding_tokens = len(doc_tokens)

        # Pad if necessary
        if len(doc_tokens) < self.max_seq_len:
            doc_tokens = np.pad(
                doc_tokens,
                (0, self.max_seq_len - len(doc_tokens)),
                mode='constant',
                constant_values=self.target_tokenizer.pad_token_id or 0
            )

        labels = doc_tokens.copy()

        # Set padding positions to -100
        if num_nonpadding_tokens < self.max_seq_len:
            labels[num_nonpadding_tokens:] = -100

        return {
            'input_ids': doc_tokens.tolist(),
            'labels': labels.tolist(),
            'text': text,
            'shard_number': shard_info['shard']['shard_number'],
            'num_nonpadding_tokens': num_nonpadding_tokens,
            'source_num_tokens': len(source_tokens),  # Original token count
        }