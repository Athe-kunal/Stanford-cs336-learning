from __future__ import annotations
import json
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        tokens_to_ids_map: dict[tuple[int, ...], int],
        ids_to_tokens_map: dict[int, list[int]],
        special_tokens: list[str],
        token_re: re.Pattern,
    ) -> None:
        self.tokens_to_ids_map = tokens_to_ids_map
        self.token_re = token_re
        self.special_tokens = special_tokens
        self.ids_to_tokens_map = ids_to_tokens_map

    def encode(self, text: str) -> list[int]:
        bytes_list: list[int] = []
        for match_ in re.finditer(self.token_re, text):
            bytes_list.extend(match_.group().encode("utf-8"))
        if not bytes_list:
            return []
        merged_list: list[int] = []
        encoded_list: list[int] = []
        for bl in bytes_list:
            merged_list.append(bl)
            if tuple(merged_list) not in self.tokens_to_ids_map:
                encoded_list.append(self.tokens_to_ids_map[tuple(merged_list[:-1])])
                merged_list = [bl]
        if tuple(merged_list) in self.tokens_to_ids_map:
            encoded_list.append(self.tokens_to_ids_map[tuple(merged_list)])
        else:
            for ml in merged_list:
                encoded_list.append(self.tokens_to_ids_map[tuple([ml])])

        return encoded_list

    def encode_iterable(self, str_iterable: Iterable[str]) -> Iterator[list[int]]:
        for iter_str in str_iterable:
            yield self.encode(iter_str)

    def decode_iterable(self, ids_iterables: Iterable[list[int]]) -> Iterator[str]:
        for id_iter in ids_iterables:
            yield self.decode(id_iter)

    def decode(self, input_ids: list[int]) -> str:
        ids_to_tokens: list[int] = []

        for iid in input_ids:
            ids_to_tokens.extend(self.ids_to_tokens_map[iid])
        byte_seq = bytes(ids_to_tokens)
        return byte_seq.decode("utf-8")

    @classmethod
    def from_files(
        cls, tokenizer_path: str, special_tokens: list[str] | None
    ) -> Tokenizer:
        with open(tokenizer_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        if not special_tokens:
            special_tokens = ["<|endoftext|>"]
        special_alt = "|".join(re.escape(s) for s in special_tokens)
        max_vocab = max([int(k) for k in list(data.keys())])
        for i in range(len(special_tokens)):
            idx = max_vocab + i
            data[str(idx)] = [idx]
        PAT = rf"""
            [ ]?(?:{special_alt})              # 1) special tags, optional leading space
        | '(?:[sdmt]|ll|ve|re)               # 2) contractions
        | [ ]?\p{{L}}+                       # 3) letters
        | [ ]?\p{{N}}+                       # 4) numbers
        | [ ]?[^\s\p{{L}}\p{{N}}]+           # 5) punctuation
        | \s+(?!\S)                          # 6) trailing spaces at EOL
        | \s+                                # 7) other whitespace
        """
        token_re = re.compile(PAT, re.VERBOSE)
        tokens_to_ids_map = {
            tuple(byte): int(token_id) for token_id, byte in data.items()
        }
        ids_to_tokens_map = {int(token_id): byte for token_id, byte in data.items()}
        return cls(
            tokens_to_ids_map=tokens_to_ids_map,
            ids_to_tokens_map=ids_to_tokens_map,
            special_tokens=special_tokens,
            token_re=token_re,
        )
