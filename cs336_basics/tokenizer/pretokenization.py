from logging import Logger
import os
from pathlib import Path
from typing import BinaryIO, Optional
import regex as re
from .chunking import find_chunk_boundaries
import multiprocessing as mp
from collections import Counter

def pretokenize_single_chunk(
    file_path: Path, start: int, end: int, pattern: re.Pattern[str], special_tokens: list[bytes], logger: Optional[Logger]
) -> dict[tuple[bytes, ...], int]:
    """pretokenize a chunk of an input file

    Args:
        file_path: input file path
        start (int): starting index
        end (int): end index (not included)
        pattern (re.Pattern): pattern for pretokenization
        logger (Optional[Logger]): (optional) logger for debug

    Returns:
        dict[tuple[bytes, ...], int]: frequencies of the pretokenized segments
    """
    if logger:
        logger.debug(f"pretokenize_single_chunk( file: {file_path}, start: {start}, end: {end})")

    with open(file_path, "rb") as file:
        # get file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # boundary check
        if start>=file_size or end > file_size or end<=start:
            return {}

        # get content string
        freq = Counter()
        file.seek(start)
        chunk_bytes :bytes = file.read(end-start)
        content = chunk_bytes.decode("utf-8")

        # split by special token
        special_token_pattern = "|".join(re.escape(special_token.decode("utf-8")) for special_token in special_tokens)
        for segment in re.split(special_token_pattern, content):
            freq.update(Counter(tuple(bytes([b]) for b in match.group().encode()) for match in re.finditer(pattern, segment)))

    return freq

def parallelized_pretokenize(
        file_path: Path, pattern: re.Pattern[str], special_tokens: list[bytes], logger:Optional[Logger]
)-> dict[tuple[bytes, ...], int]:
    """parellelized pretokenization process

    Args:
        file_path (Path): data path
        pattern (re.Pattern[str]): regex pattern for pretokenization
        special_tokens (list[bytes]): special token list
        logger (Optional[Logger]): logger

    Returns:
        dict[bytes, int]: frequency of pre-tokenized strings
    """
    with open(file_path, "rb") as file:
        boundaries:list[int] = find_chunk_boundaries(file, 16)
        chunks: list[bytes] = []
        tasks = []
        pre_tokenized_text = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((file_path, start, end, pattern, special_tokens, logger))
        if logger:
            logger.info(f"tasks chunking completed")
        with mp.Pool(processes=int(os.getenv("NUM_WORKERS", 4))) as pool:
            results = pool.starmap(pretokenize_single_chunk, tasks)
            result = Counter()
            for i in results:
                result.update(i)
            return result