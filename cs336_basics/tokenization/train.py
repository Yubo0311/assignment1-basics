import logging
from re import DEBUG
from typing import Optional
from chunking import find_chunk_boundaries
import multiprocessing as mp
import os
import regex as re
from pretokenization import pretokenize_single_chunk, parallelized_pretokenize
from pathlib import Path

# configure logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# pre-tokenization pattern definition
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# special tokens definition
SPECIAL_TOKENS = [b"<|endoftext|>"]

logger = logging.getLogger(__name__)

def train_bpe(file_path: Path, vocab_size: int, special_tokens: list[bytes], logger: Optional[logging.Logger]) ->dict[tuple[bytes, ...], int]:
    """train a bpe tokenizer
    Args: 
        input_path: Path to the training corpus
        vocab_size: A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary.
    Return:
        vocab: The tokenizer vocabulary, a mapping from int (token ID) to bytes.
        merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1> , <token2>).
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, ...]] = []
    try:
        # 0. Open and read dataset
        with open(file_path, "rb") as f:
            if logger:
                logger.info(f"Successfully opened file: {file_path} !")
            # 1. Initialize vocabulary
            for i in range(256):
                vocab[i] = bytes([i])
            for i, special_token in enumerate(special_tokens):
                vocab[256+i] = special_token
            # 2. Parallelized Pre-tokenization
            pre_tokenized = parallelized_pretokenize(file_path, PAT, SPECIAL_TOKENS, logger)
            return pre_tokenized
    except FileNotFoundError:
        print(f"file didn't exist: {file_path}")
    except PermissionError:
        print(f"no permission to read file: {file_path}")
    return {}

def readable_freq(freq: dict[tuple[bytes, ...], int])->dict[str, int]:
    return {bytes(i).decode():j for i, j in freq.items()}

if __name__ == "__main__":
    print(readable_freq(train_bpe(Path("data/owt_valid.txt"), 256, SPECIAL_TOKENS, logger)))






