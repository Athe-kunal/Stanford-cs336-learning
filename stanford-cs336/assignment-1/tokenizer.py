import collections
import concurrent.futures
import regex as re
import json
import os
from tqdm import tqdm
from typing import TypeAlias, Generator, Any

BASE_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)
TXT_FILE = "/Users/athekunal/Desktop/Stanford-cs336/Stanford-cs336-learning/stanford-cs336/data/TinyStoriesV2-GPT4-valid.txt"

BytesTuple: TypeAlias = tuple[int, ...]
BytesCount: TypeAlias = int

VocabDict: TypeAlias = collections.defaultdict[BytesTuple, BytesCount]
MergesDict: TypeAlias = list[tuple[int, ...]]
TokenID: TypeAlias = int


def convert_to_bytes(
    text: str,
) -> collections.defaultdict[BytesTuple, BytesCount]:
    bytes_dict: collections.defaultdict[BytesTuple, BytesCount] = (
        collections.defaultdict(int)
    )
    for match in re.finditer(_PAT, text):
        bytes_key = tuple(match.group().encode("utf-8"))
        bytes_dict[bytes_key] += 1
    return bytes_dict


def read_until_eot(filepath, delimiter="<|endoftext|>", chunk_size=1024):
    buffer = ""
    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer:
                    yield buffer  # last chunk (no delimiter at end)
                break

            buffer += chunk

            while delimiter in buffer:
                idx = buffer.index(delimiter)
                yield buffer[:idx]
                buffer = buffer[idx + len(delimiter) :]


def replace_all_subsequences(
    a: tuple[int, ...], b: tuple[int, ...], curr_max_vocab: int
) -> tuple[int, ...]:
    if not b:
        return a
    n, m = len(a), len(b)
    result = []
    i = 0
    while i <= n - m:
        if a[i : i + m] == b:
            result.append(curr_max_vocab)
            i += m
        else:
            result.append(a[i])
            i += 1
    result.extend(a[i:])
    return tuple(result)


def _count_successive_pairs(bytes_key: BytesTuple, bytes_cnt: BytesCount) -> VocabDict:
    aggr_successive_dict: VocabDict = collections.defaultdict(int)
    for i in range(len(bytes_key) - 1):
        aggr_successive_dict[(bytes_key[i], bytes_key[i + 1])] += bytes_cnt
    return aggr_successive_dict


def _replace_and_keep_cnt(
    k: BytesTuple, top1: BytesTuple, curr_max_vocab: int, cnt: int
) -> VocabDict:
    new_key = replace_all_subsequences(a=k, b=top1, curr_max_vocab=curr_max_vocab)
    result = collections.defaultdict(int)
    result[new_key] = cnt
    return result


# at module scope
def _count_successive_pairs_wrapper(arg):
    bytes_key, bytes_cnt = arg
    return _count_successive_pairs(bytes_key, bytes_cnt)


def _replace_and_keep_wrapper(arg):
    k, cnt, best_pair, new_id = arg
    return replace_all_subsequences(k, best_pair, new_id), cnt


def get_chunksize(items: list[Any]) -> int:
    total_cpu_count = min(os.cpu_count() or 4, len(items))
    chunk_size = len(items) // total_cpu_count
    return max(1, chunk_size)


def get_new_vocab_dict(
    curr_vocab_dict: VocabDict, curr_max_vocab: int
) -> tuple[VocabDict, tuple[int, ...]]:
    successive_dict: VocabDict = collections.defaultdict(int)
    items = list(curr_vocab_dict.items())
    chunk_size = get_chunksize(items=items)
    args_iter_pairs = ((bk, bcnt) for bk, bcnt in items)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for partial in executor.map(
            _count_successive_pairs_wrapper, args_iter_pairs, chunksize=chunk_size
        ):
            for pair, cnt in partial.items():
                successive_dict[pair] += cnt

    top1, top1_count = max(successive_dict.items(), key=lambda item: item[1])
    combined_vocab_dict: VocabDict = collections.defaultdict(int)

    args_iter = ((k, v, top1, curr_max_vocab) for k, v in items)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for new_key, cnt in executor.map(
            _replace_and_keep_wrapper, args_iter, chunksize=chunk_size
        ):
            combined_vocab_dict[new_key] += cnt
    return combined_vocab_dict, top1


def unroll_merges(
    curr_vocab_dict: collections.defaultdict[TokenID, BytesTuple],
    curr_merges: tuple[int, ...],
    initial_vocab_len: int,
) -> tuple[int, ...]:
    actual_token_ids: list[int] = []

    for cm in curr_merges:
        assert cm in curr_vocab_dict, f"Could not find the token id {cm}"
        if cm >= initial_vocab_len:
            token_ids = curr_vocab_dict[cm]
            actual_token_ids.extend(token_ids)
        else:
            actual_token_ids.append(cm)
    return tuple(actual_token_ids)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    delimiter: str = "<|endoftext|>",
    chunk_size: int = 1024,
) -> tuple[collections.defaultdict[TokenID, BytesTuple], MergesDict, dict[str, int]]:
    initial_vocab = list(range(BASE_VOCAB_SIZE)) + [
        BASE_VOCAB_SIZE + i for i in range(len(special_tokens))
    ]
    initial_vocab_size = BASE_VOCAB_SIZE + len(special_tokens)
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} is too small; must be at least {initial_vocab_size}"
        )
    vocab_dict: collections.defaultdict[TokenID, BytesTuple] = collections.defaultdict(
        tuple
    )
    for iv in initial_vocab:
        vocab_dict[iv] = (iv,)
    special_tokens_map = {
        st: BASE_VOCAB_SIZE + i for i, st in enumerate(special_tokens)
    }
    num_merges = vocab_size - len(initial_vocab)
    all_bytes_dict: VocabDict = collections.defaultdict(int)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                convert_to_bytes,
                read_until_eot(
                    filepath=input_path, delimiter=delimiter, chunk_size=chunk_size
                ),
            )
        )
        for res in results:
            for key, val in res.items():
                all_bytes_dict[key] += val
    merges: list[tuple[int, ...]] = []
    for merge in tqdm(range(num_merges), desc=f"Running for merges {num_merges}"):
        curr_max_vocab = BASE_VOCAB_SIZE + len(special_tokens) + merge
        all_bytes_dict, curr_merges = get_new_vocab_dict(
            all_bytes_dict, curr_max_vocab=curr_max_vocab
        )
        vocab_dict[curr_max_vocab] = unroll_merges(
            vocab_dict, curr_merges, initial_vocab_size
        )
        merges.append(curr_merges)

    return vocab_dict, merges, special_tokens_map


def save_dict_to_json(data: dict, file_path: str, indent: int = 4) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    vocab_dict, merges, special_tokens_map = train_bpe(TXT_FILE, 500, ["<|endoftext|>"])
    save_dict_to_json(vocab_dict, "temp2.json", indent=1)
