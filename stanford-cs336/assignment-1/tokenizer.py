import collections
import concurrent.futures
import regex as re
from typing import TypeAlias

BytesTuple: TypeAlias = tuple[int, ...]
BytesCount: TypeAlias = int
VocabDict: TypeAlias = collections.defaultdict[BytesTuple, BytesCount]
TokenID: TypeAlias = int

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
txt_file = "/Users/athekunal/Desktop/Stanford-cs336/Stanford-cs336-learning/stanford-cs336/data/TinyStoriesV2-GPT4-valid.txt"


def convert_to_bytes(
    text: str,
) -> collections.defaultdict[BytesTuple, BytesCount]:
    bytes_dict: collections.defaultdict[BytesTuple, BytesCount] = collections.defaultdict(
        int
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

def get_new_vocab_dict(vocab_dict: VocabDict, curr_max_vocab: int) -> VocabDict:
    successive_dict: VocabDict = collections.defaultdict(int)
    for k, v in vocab_dict.items():
        for i in range(len(k)-1):
            successive_dict[(k[i],k[i+1])]+=v
    top1 = max(successive_dict.items(), key=lambda item: item[1])
    combined_vocab_dict: VocabDict = collections.defaultdict(int)
    for k,v in vocab_dict.items():
        

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    delimiter: str = "<|endoftext|>",
    chunk_size: int = 1024,
) -> None:
    # ) -> tuple[collections.defaultdict[int, int], list[tuple[bytes, bytes]]]:
    initial_vocab = list(range(256)) + [256 + i for i in range(len(special_tokens))]
    vocab_dict: collections.defaultdict[TokenID, BytesTuple] = collections.defaultdict(BytesTuple)
    for iv in initial_vocab:
        vocab_dict[iv] = (iv,)
    special_tokens_map = {st: 256 + i for i, st in enumerate(special_tokens)}
    num_merges = vocab_size - len(initial_vocab)
    all_bytes_dict: VocabDict = (
        collections.defaultdict(int)
    )
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

    for merge in range(num_merges):



if __name__ == "__main__":
    train_bpe(txt_file, 100, ["endoftext"])
