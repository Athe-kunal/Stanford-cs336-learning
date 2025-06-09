from __future__ import annotations
import json
import regex as re
from typing import Iterable


class Tokenizer:
    def __init__(
        self,
        tokenizer_map: dict[tuple[int, ...], int],
        special_tokens: list[str],
        token_re: re.Pattern,
    ) -> None:
        self.tokenizer_map = tokenizer_map
        self.token_re = token_re
        self.special_tokens = special_tokens

    # def encode(self, text: str) -> list[int]:
    #     bytes_list: list[int] = []
    #     for match_ in re.finditer(self.token_re, text):
    #         bytes_list.extend(match_.group().encode("utf-8"))
    #     token_ids: list[int] = []
    #     aggr_ids: list[int] = []
    #     temp_ids: list[int] = []
    #     for bl in bytes_list:
    #         if (bl) in self.tokenizer_map

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
        print(data)
        tokenizer_map = {tuple(byte): token_id for token_id, byte in data.items()}
        return cls(
            tokenizer_map=tokenizer_map,
            special_tokens=special_tokens,
            token_re=token_re,
        )
