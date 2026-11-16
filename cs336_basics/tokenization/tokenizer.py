import regex as re
from typing import Iterator, Iterable
class tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_to_id = {value:key for key, value in vocab.items()}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def merge_bytes(self, l:list[bytes],i: int):
        """helper function to merge adjacent bytes in a list

        Args:
            l (list[bytes]): the list of bytes
            i (int): the index of the first bytes to merge
        """
        assert i+1<len(l)
        l[i] = b"".join([l[i], l.pop(i+1)])

    def encode_partial(self, s:str)->list[int]:
        """helper function to encode a part without special tokens

        Args:
            s (str): input string
            start (int): start index
            end (int): end

        Returns:
            list[int]: ids
        """
        s_bytes = s.encode("utf-8")
        s_bytes_list = [s_bytes[i:i+1] for i in range(len(s_bytes))]
        for merge in self.merges:
            i = 0
            while i < len(s_bytes_list)-1:
                if merge==(s_bytes_list[i], s_bytes_list[i+1],):
                    self.merge_bytes(s_bytes_list, i)
                i+=1
        s_id_list = [self.vocab_to_id[i] for i in s_bytes_list]
        return s_id_list



    def encode(self, s:str)->list[int]:
        """encode a string

        Args:
            s (str): string to be encoded

        Returns:
            list[int]: result, a list of vocabulary ids
        """
        if self.special_tokens:
            # We sort the special token list so that the longer ones are before shorter ones
            # in case that there are one special token containing another
            special_token_pattern =re.compile( "("+"|".join(sorted([re.escape(special_token) for special_token in self.special_tokens], key = lambda x: -len(x)))+")")
        else:
            return self.encode_partial(s)
        result = []
        for i, segment in enumerate(re.split(special_token_pattern, s)):
            if i%2==0:
                for pretoken in re.findall(self.PAT, segment):
                    # not special token
                    result.extend(self.encode_partial(pretoken))
            else:
                # in split result, there might be empty string "" as placeholder
                if segment:
                    result.append(self.vocab_to_id[segment.encode("utf-8")])
        return result

        

    def decode(self, ids:list[int])->str:
        """decode a list of vocab ids

        Args:
            ids (list[int]): list of vocab ids

        Returns:
            str: the result string
        """
        return (b"".join(self.vocab[id] for id in ids)).decode("utf-8","replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for segment in iterable:
            yield from self.encode(segment)