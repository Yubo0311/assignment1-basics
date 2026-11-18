from collections import Counter
from queue import PriorityQueue
import logging
from typing import Optional
import tqdm
from .tokenizer import Tokenizer

def update_freq(freq:dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes], candidate_keys:set[tuple[bytes, ...]], logger: Optional[logging.Logger]) -> dict[tuple[bytes, ...], int]:
    """update frequency table, merging the input pair

    Args:
        freq (dict[tuple[bytes, ...], int]): previous freq table
        pair (tuple[bytes, bytes]): bytes to be merged
        logger (Optional[logging.Logger]): logger

    Returns:
        dict[tuple[bytes, ...], int]: updated freq table
    """
    # There is possibly multiple matching pairs in a segment 
    new_freq = {}
    pair_bytes = b"".join(pair)
    for key in candidate_keys:
        if key not in freq:
            continue
        # scan the whole segment
        head = 0
        tail = 0
        new_key_list: list[bytes] = []
        prefix = False
        while(True):
            if tail>=len(key):
                if head<len(key):
                    new_key_list.extend(key[head:tail])
                break
            elif tail<len(key)-1 and key[tail:tail+2] == pair:
                # found pair, split and flush
                if tail-head>0:
                    new_key_list.extend(key[head:tail])
                new_key_list.append(pair_bytes)
                tail+=2
                head = tail
            else:
                tail+=1
        new_key = tuple(new_key_list)
        freq[new_key] = new_freq.get(new_key, 0)+freq[key]
        freq.pop(key)
            

    if logger is not None:
        logger.debug("Merging pair: %r -> %r", pair, pair_bytes)

        for key, value in freq.items():
            if key not in new_freq:
                logger.debug("REMOVE: %r : %d", key, value)

        for key, value in new_freq.items():
            if key not in freq:
                logger.debug("ADD:    %r : %d", key, value)
            elif freq[key] != value:
                logger.debug(
                    "UPDATE: %r : %d -> %d",
                    key, freq[key], value
                )

    
def bpe(vocab_size:int, special_tokens: list[bytes], freq:dict[tuple[bytes, ...], int],  logger:Optional[logging.Logger]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """bpe tokenizer

    Args:
        vocab_size (int): the desired vocabulary size
        special_tokens (list[bytes]): special tokens
        freq (dict[tuple[bytes, ...], int]): frequencies of segments from pre-tokenization
        logger (Optional[logging.Logger]): logger

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: the id-to-token map and merges sequence
    """
    # initialize vocabulary
    vocab:dict[int, bytes] = {}
    vocab_to_id:dict[bytes, int] = {}
    for i in range(256):
        vocab[i] = bytes([i])
        vocab_to_id[bytes([i])] = i
    for i, special_token in enumerate(special_tokens):
        vocab[256+i] = special_token
        vocab_to_id[special_token] = 256+i
    if logger:
        logger.debug(f"vocabulary initialized! Size: {len(vocab)}")

    merges = []


    # start 
    pair_to_segment : dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for id in range(len(vocab), vocab_size):
        pair_freqs = Counter()
        # count byte pair
        for key, value in freq.items():
            for pair in zip(key[:-1], key[1:]):
                pair_freqs[pair] += value
                if pair in pair_to_segment:
                    pair_to_segment[pair].add(key)
                else:
                    pair_to_segment[pair] = {key}

        # select most frequent pair, if tie, select the lexically greater one
        max_pair= max(pair_freqs, key = lambda x: (pair_freqs[x], x))

        merges.append(max_pair)
        pair_bytes = b"".join(max_pair)

        vocab[id] = pair_bytes
        vocab_to_id[pair_bytes] = id

        
        update_freq(freq, max_pair, pair_to_segment[max_pair], logger)
    
    return (vocab, merges)

    