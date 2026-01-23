import os
import json
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch
import logging
import zstandard as zstd
import json
import argparse
# import tiktoken



def read_jsonl_zst(file_path):
    """WebOrganizer corpus is big. Want to use streaming to avoid loading all into memory."""

    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**14)  # 16KB at a time
                if not chunk:
                    break
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        yield json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue


def read_weborg_shard(shard_path, n_docs=None):
    """Read a WebOrganizer shard. Stop after n_docs if specified."""
    n_docs_read = 0
    for obj in read_jsonl_zst(shard_path):
        if n_docs is not None and n_docs_read > n_docs:
            break
        n_docs_read += 1
        yield obj



def tokenize_and_save_documents_weborg(shard_path, output_dir, tokenizer, split_name, shard_number, num_docs=None, max_length=2048):

    # Save arrays
    tokens_file = os.path.join(output_dir, f"{split_name}_{shard_number}_tokens.npy")
    boundaries_file = os.path.join(output_dir, f"{split_name}_{shard_number}_boundaries.npy")

    if os.path.exists(tokens_file) and os.path.exists(boundaries_file):
        print(f"Skipping {shard_path} because it already exists")
        return

    all_tokens = []
    doc_boundaries = [0]  # Start of first document

    for doc in read_weborg_shard(shard_path, n_docs=num_docs):
        # Tokenize
        tokens = tokenizer(doc["text"], add_special_tokens=True)['input_ids']

        # Truncate to max length
        tokens = tokens[:max_length]

        all_tokens.extend(tokens)

        # Record end of this document (start of next document)
        doc_boundaries.append(len(all_tokens))

    # Convert to numpy arrays
    tokens_array = np.array(all_tokens, dtype=np.int64)
    boundaries_array = np.array(doc_boundaries[:-1], dtype=np.int64)  # Remove last boundary


    np.save(tokens_file, tokens_array)
    np.save(boundaries_file, boundaries_array)

    print(f"Saved {split_name} split:")
    print(f"  - Tokens: {len(tokens_array):,} tokens in {tokens_file}")
    print(f"  - Documents: {len(boundaries_array):,} documents in {boundaries_file}")


def _main(args):
    # Tokenize the SlimPajama corpus

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 


    corpus_dir = "/Corpus-200B/documents"
    shard_paths = sorted(os.listdir(corpus_dir))

    np.random.seed(42)
    shard_paths = np.random.permutation(shard_paths)

    # split into train, val, test.
    # train: 0.0 -> 0.5
    # cooldown: 0.50 -> 0.70
    # val: 0.7 -> 0.85
    # test: 0.85 -> 1.0
    # shard_paths_train = shard_paths[:int(len(shard_paths)*0.03)]
    shard_paths_cooldown = shard_paths[int(len(shard_paths)*0.5):int(len(shard_paths)*0.51)]
    # shard_paths_val = shard_paths[int(len(shard_paths)*0.7):int(len(shard_paths)*0.702)]
    # shard_paths_test = shard_paths[int(len(shard_paths)*0.85):int(len(shard_paths)*0.854)]

    # for shard_paths, split_name in zip([shard_paths_train, shard_paths_cooldown, shard_paths_val, shard_paths_test], ['train', 'cooldown', 'val', 'test']):
    for shard_paths, split_name in zip([shard_paths_cooldown], ['cooldown']): 

        # Make folder for tokenized data
        folder_path = os.path.join(args.output_dir, args.tokenizer_name, split_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        for shard_path in tqdm(shard_paths):
            # get shard number
            shard_number = shard_path.split('_')[2]

            tokenize_and_save_documents_weborg(
                os.path.join(corpus_dir, shard_path), 
                folder_path, 
                tokenizer, 
                split_name=split_name, 
                shard_number=shard_number
            )

def main():

    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output_dir", type=str,
                        help="Path to config file")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                        help="Name of the tokenizer to use.")
    parser.add_argument("--test_mode", action='store_true', default=False,
                        help="Test mode. Only tokenize a subset of the corpus.")

    # Pipeline arguments
    parser.add_argument("--redpajama", action='store_true', default=False,
                        help="Tokenize the RedPajama corpus.")
    parser.add_argument("--weborganizer", action='store_true', default=False,
                        help="Tokenize the WebOrganizer corpus.") 

    parser.add_argument("--num_docs", type=int, default=None,   
                        help="Number of documents to tokenize. If not specified, tokenize the entire corpus. For WebOrg only.")
            




    args = parser.parse_args()
    return _main(args)

if __name__ == "__main__":
    main()