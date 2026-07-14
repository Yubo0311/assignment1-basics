from typing import BinaryIO
from sys import maxsize
import logging
import re
import multiprocessing
import os

GPT2_REGEX_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"
SPECIAL_TOKENS: dict[str, bytes] = {"endoftext": b"<|endoftext|>"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
def find_chunk_boundaries_group(
    file: BinaryIO,
    desired_num_chunks: int = 4,
    split_special_tokens: tuple[bytes,...] = (SPECIAL_TOKENS["endoftext"],),
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = maxsize
            # Find the special token in the mini chunk
            for special_token in split_special_tokens:
                new_pos = mini_chunk.find(special_token)
                if(new_pos==-1):
                    pass
                else:
                    found_at = min(found_at, new_pos)
            if found_at != maxsize:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int = 4,
    split_special_token: bytes = SPECIAL_TOKENS["endoftext"],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    logger.debug(f"file_size = {file_size}")
    logger.debug(f"chunk_size = {chunk_size}")
    logger.debug(f"special_token = {split_special_token.decode()}")

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def serialized_pre_tokenize(chunk: str):
    pass

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    merges: list[tuple[bytes, bytes]] = []
    try:
        # 0. Open and read dataset
        with open(input_path, "rb") as f:
            # 1. Initialize vocabulary
            for i in range(255):
                vocab[i] = bytes([i])
            vocab[256] = SPECIAL_TOKENS["endoftext"]
            nextid: int = 257;
            # 2. Parallelized Pre-tokenization
            boundaries:list[int] = find_chunk_boundaries(f)
            chunks: list[bytes] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end-start).decode("utf-8", errors="ignore")
                logger.debug(chunk[-100:])
                logger.debug("================ SEPARATOR ===================")
                logger.debug("")

            

            


    except FileNotFoundError:
        print(f"file didn't exist: {input_path}")
    except PermissionError:
        print(f"no permission to read file: {input_path}")


    return vocab, merges

train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])