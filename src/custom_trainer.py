
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
from torch.utils.data import Dataset as TorchDataset, DataLoader, Sampler
from datasets import Dataset
import os
import logging
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from collections import defaultdict
import random
from tqdm import tqdm

class EpochCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(state.epoch)
                    

class ShardBatchSampler(Sampler):
    """Custom sampler that creates batches where all samples come from the same shard."""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by shard
        self.shard_groups = defaultdict(list)
        for idx in range(len(dataset)):
            # Get the shard number for this index
            doc_idx = dataset.shuffled_indices[idx]
            shard_idx = None
            for i, info in enumerate(dataset.shard_info):
                if info['start_doc_idx'] <= doc_idx < info['end_doc_idx']:
                    shard_idx = i
                    break
            if shard_idx is not None:
                self.shard_groups[shard_idx].append(idx)
    
    def __iter__(self):
        # Create batches within each shard
        all_batches = []
        
        for shard_idx, indices in self.shard_groups.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from this shard
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle the order of batches if needed
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for indices in self.shard_groups.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


class ShardAwareDataCollator:
    """Data collator that preserves shard information and ensures batch consistency."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Check that all features come from the same shard
        shard_numbers = [f['shard_number'] for f in features]
        if len(set(shard_numbers)) > 1:
            raise ValueError(f"Batch contains samples from multiple shards: {set(shard_numbers)}")
        
        batch_shard_number = shard_numbers[0]
        
        # Prepare the batch
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long),
            'shard_number': batch_shard_number,  # Add shard info to batch
            'num_nonpadding_tokens': sum([f['num_nonpadding_tokens'] for f in features]),
            'num_nonpadding_tokens_array': [f['num_nonpadding_tokens'] for f in features],
        }
        
        # Correct attention mask: 1 for real tokens, 0 for padding (token_id = 0)
        batch['attention_mask'] = (batch['input_ids'] != 0).long()

        return batch

class CustomTrainer(Trainer):

    """Custom Trainer that saves per-token losses during evaluation."""

    def __init__(self, eval_save_dir=None, max_seq_len=2048, eval_only=False, use_shard_batching=True, is_cooldown=False, test_mode=False, overwrite_eval_losses=True, tokenizer=None, **kwargs):
        self.use_shard_batching = use_shard_batching
        super().__init__(**kwargs)
        self.eval_save_dir = eval_save_dir
        self.eval_step_count = 0
        self.max_seq_len = max_seq_len
        self.eval_only = eval_only
        self.is_cooldown = is_cooldown
        self.test_mode = test_mode
        self.overwrite_eval_losses = overwrite_eval_losses
        self.tokenizer = tokenizer

    def get_train_dataloader(self):
        """Override to use custom shard-aware sampler if enabled."""
        if not self.use_shard_batching:
            return super().get_train_dataloader()
        
        
        sampler = ShardBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use custom shard-aware sampler if enabled."""
        if not self.use_shard_batching:
            return super().get_eval_dataloader(eval_dataset)
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        sampler = ShardBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,  # Don't shuffle eval data
            drop_last=False
        )
        
        return DataLoader(
            eval_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to save per-token losses."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset


        # Standard evaluation
        metrics = {}
        
        # Save per-token losses if eval_save_dir is provided
        if self.eval_only:
            assert self.eval_save_dir and hasattr(self, 'model')
            self._save_per_token_losses(eval_dataset)
        else:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    

        return metrics


    def _save_per_token_losses(self, eval_dataset):
        """Save per-token losses to file."""

        assert self.eval_save_dir is not None, "eval_save_dir must be provided"
        
        self.model.eval()
        shard_losses = []
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        with torch.no_grad():
            first_batch = True
            total_nonpadding_tokens = 0
            count = 0

            for batch in tqdm(eval_dataloader):
                _batch = batch.copy()

                if first_batch:
                    first_batch = False
                    prev_shard = batch['shard_number']

                # Save losses once new shard is encountered
                if prev_shard != batch['shard_number']:

                    # Convert shard_losses into one large flat array
                    shard_losses = np.concatenate(shard_losses, axis=0)
                    assert shard_losses.shape[0] == total_nonpadding_tokens, f"{shard_losses.shape[0]} != {total_nonpadding_tokens}!!"

                    os.makedirs(self.eval_save_dir, exist_ok=True)
        
                    # Use global step if available, otherwise use eval count
                    eval_file = os.path.join(self.eval_save_dir, f"eval_losses_shard_{prev_shard}.npy")
                    
                    #Todo: get the right save directory
                    np.save(eval_file, shard_losses)


                    logging.info(f"Saved per-token losses to {eval_file}")
                    
                    logging.info(f"Shape: {shard_losses.shape}, Mean loss: {np.mean(shard_losses):.4f}")

                    shard_losses = []
                    prev_shard = batch['shard_number']
                    total_nonpadding_tokens = 0

                
                
                # Move to device
                batch = self._prepare_inputs(batch)
                num_nonpadding_tokens = batch.pop('num_nonpadding_tokens')
                num_nonpadding_tokens_array = batch.pop('num_nonpadding_tokens_array')
                total_nonpadding_tokens += num_nonpadding_tokens

                # Remove shard number from batch
                batch.pop('shard_number')

                # Forward pass
                outputs = self.model(**batch)

                # Get per-token losses
                logits = outputs.logits
                labels = batch['labels']

                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Get attention mask for proper masking (SFTTrainer should provide this)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    # Shift attention mask to match labels
                    shift_attention_mask = attention_mask[..., 1:].contiguous()
                else:
                    # Create default attention mask if not provided
                    shift_attention_mask = torch.ones_like(shift_labels)

                # Compute cross entropy for each token
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # This mask is True only where the value is -0.0
                negzero_mask = (token_losses == 0) & torch.signbit(token_losses)

                # Replace -0.0 with +0.0
                token_losses = torch.where(negzero_mask, torch.full_like(token_losses, 1e-12), token_losses)

                if self.test_mode and count == 20:
                    import pdb; pdb.set_trace()
                # Reshape back and apply attention mask
                token_losses = token_losses.view(shift_labels.shape)

                # Mask out padded tokens - only keep losses where attention_mask is 1
                # and labels are not -100
                valid_mask = (shift_labels != -100) & (shift_attention_mask == 1)
                masked_losses = torch.where(valid_mask, token_losses, torch.zeros_like(token_losses))
                np_masked_losses = masked_losses.cpu().numpy()
                

                # Add a single np.nan to each of the sequences (since we don't have loss over last token)
                sequence_length = np_masked_losses.shape[1]
                np_masked_losses = np.pad(masked_losses.cpu().numpy(), 
                         ((0, 0), (0, 1)), 
                         mode='constant', 
                         constant_values=np.nan)

                assert np_masked_losses.shape[1] == sequence_length + 1


                arr = []
                for subarray, _num_nonpadding_tokens in zip(np_masked_losses, _batch['num_nonpadding_tokens_array']):
                    subarray = subarray[:_num_nonpadding_tokens]
                    arr.append(subarray)
            
                # make arr one big array
                reshaped_losses = np.concatenate(arr, axis=0)

                if len(reshaped_losses) != num_nonpadding_tokens:
                    import pdb; pdb.set_trace()
                    assert False, "Reshaped losses are not equal to num_nonpadding_tokens"
                
                shard_losses.append(reshaped_losses)

                count += 1
                if self.test_mode and count > 20:
                    import pdb; pdb.set_trace()

        # Save last shard
        shard_losses = np.concatenate(shard_losses, axis=0)
        assert shard_losses.shape[0] == total_nonpadding_tokens, f"{shard_losses.shape[0]} != {total_nonpadding_tokens}!!"


        os.makedirs(self.eval_save_dir, exist_ok=True)

        # Use global step if available, otherwise use eval count
        eval_file = os.path.join(self.eval_save_dir, f"eval_losses_shard_{prev_shard}.npy")
        
        #Todo: get the right save directory
        np.save(eval_file, shard_losses)


        logging.info(f"Saved per-token losses to {eval_file}")
        
        logging.info(f"Shape: {shard_losses.shape}, Mean loss: {np.mean(shard_losses):.4f}")
        logging.info(f"Done evaluating!")
            
        self.eval_step_count += 1
