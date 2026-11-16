import logging
from re import DEBUG
from tokenize import Special
from typing import Optional
from .chunking import find_chunk_boundaries
import multiprocessing as mp
import os
import regex as re
from .pretokenization import pretokenize_single_chunk, parallelized_pretokenize
from pathlib import Path
from .bpe import bpe
import argparse

# configure logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# pre-tokenization pattern definition
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# special tokens definition
SPECIAL_TOKENS = [b"<|endoftext|>"]

logger = logging.getLogger(__name__)

def train_bpe(file_path: Path, vocab_size: int, special_tokens : list[bytes] = SPECIAL_TOKENS, logger: Optional[logging.Logger] = None) ->tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :
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
        with open(file_path, "rb") as f:
            if logger:
                logger.info(f"Successfully opened file: {file_path} !")
            pre_tokenized = parallelized_pretokenize(file_path, PAT, SPECIAL_TOKENS, logger)
            return bpe(vocab_size, special_tokens, pre_tokenized, logger)

    except FileNotFoundError:
        print(f"file didn't exist: {file_path}")
    empty_result:tuple[dict[int, bytes], list[tuple[bytes, bytes]]] = ({}, []) 
    return empty_result

def readable_freq(freq: dict[tuple[bytes, ...], int])->dict[str, int]:
    return {b"".join(i).decode():j for i, j in freq.items()}






