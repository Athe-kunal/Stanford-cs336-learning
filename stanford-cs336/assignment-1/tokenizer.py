import collections
import concurrent.futures
import regex as re
from tqdm import tqdm
from typing import TypeAlias


BytesTuple: TypeAlias = tuple[int, ...]
BytesCount: TypeAlias = int

VocabDict: TypeAlias = collections.defaultdict[BytesTuple, BytesCount]
MergesDict: TypeAlias = list[tuple[int, ...]]
TokenID: TypeAlias = int
BASE_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
txt_file = "/Users/athekunal/Desktop/Stanford-cs336/Stanford-cs336-learning/stanford-cs336/data/TinyStoriesV2-GPT4-valid.txt"


def convert_to_bytes(
    text: str,
) -> collections.defaultdict[BytesTuple, BytesCount]:
    bytes_dict: collections.defaultdict[BytesTuple, BytesCount] = (
        collections.defaultdict(int)
    )
    for match in list(re.finditer(PAT, text)):
        bytes_key = tuple(list(match.group().encode("utf-8")))
        # group by two successive bytes and make the count, to reduce one less iteration
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


def get_new_vocab_dict(
    curr_vocab_dict: VocabDict, curr_max_vocab: int
) -> tuple[VocabDict, tuple[int, ...]]:
    successive_dict: collections.defaultdict[tuple[int, ...], int] = (
        collections.defaultdict(int)
    )
    for k, v in curr_vocab_dict.items():
        for i in range(len(k) - 1):
            successive_dict[(k[i], k[i + 1])] += v

    top1 = max(successive_dict.items(), key=lambda item: item[1])
    combined_vocab_dict: VocabDict = collections.defaultdict()
    for k, v in curr_vocab_dict.items():
        new_key = replace_all_subsequences(k, top1[0], curr_max_vocab=curr_max_vocab)
        combined_vocab_dict[new_key] = v
    return combined_vocab_dict, top1[0]


def unroll_merges(
    vocab_dict: collections.defaultdict[TokenID, BytesTuple],
    curr_merges: tuple[int, ...],
    initial_vocab_len: int,
) -> tuple[int, ...]:
    actual_token_ids: list[int] = []

    for cm in curr_merges:
        assert cm in vocab_dict, f"Could not find the token id {cm}"
        if cm >= initial_vocab_len - 1:
            token_ids = vocab_dict[cm]
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
    vocab_dict: collections.defaultdict[TokenID, BytesTuple] = collections.defaultdict(
        BytesTuple
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
    for merge in tqdm(range(num_merges)):
        curr_max_vocab = BASE_VOCAB_SIZE + len(special_tokens) + merge
        all_bytes_dict, curr_merges = get_new_vocab_dict(
            all_bytes_dict, curr_max_vocab=curr_max_vocab
        )
        vocab_dict[curr_max_vocab] = unroll_merges(
            vocab_dict, curr_merges, len(initial_vocab)
        )
        merges.append(curr_merges)

    return vocab_dict, merges, special_tokens_map


if __name__ == "__main__":
    vocab_dict, merges, special_tokens_map = train_bpe(txt_file, 500, ["<|endoftext|>"])
    print(vocab_dict)
    print(merges)
