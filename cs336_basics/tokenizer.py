
import os
import timeit
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import numpy.typing as npt
import numpy as np
from pathlib import Path

import regex as re

from .load_data import load_text_from_file
from .data_structure import MaxHeap


################################################################################
# Part I: Incremental BPE Training Algorithm
################################################################################

def pre_tokenize_train(
    chunks: Iterable[str],
    special_tokens: list[str],
    token_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter:
    special_pattern = re.compile('|'.join(map(re.escape, special_tokens)))
    token_pattern = re.compile(token_pattern, re.UNICODE)
    return Counter(
        match.group().encode("utf-8")
        for chunk in chunks
        for text in special_pattern.split(chunk) if text
        for match in token_pattern.finditer(text) if match.group()
    )


def pre_tokenize_infer(
    chunks: Iterable[str],
    special_tokens: list[str] | None,
    token_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Iterator[bytes]:
    # If special_tokens is None, no need to capture them
    has_special_tokens = special_tokens is not None
    if has_special_tokens:
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = re.compile(
            '(' + '|'.join(map(re.escape, special_tokens)) + ')')
    token_pattern = re.compile(token_pattern, re.UNICODE)
    for chunk in chunks:
        texts = special_pattern.split(chunk) if has_special_tokens else [chunk]
        for text in texts:
            if not text:
                continue
            if special_tokens is not None and text in special_tokens:
                yield text.encode("utf-8")
            else:
                for match in token_pattern.finditer(text):
                    if not match.group():
                        continue
                    yield match.group().encode("utf-8")


class IndexManager:
    def __init__(self):
        self.total = 0
        self.indices: set[tuple[int, int]] = set()

    def add(self, p_id, b_id, num):
        self.total += num
        self.indices.add((p_id, b_id))
        return self.total

    def remove(self, p_id, b_id, num):
        self.total -= num
        self.indices.remove((p_id, b_id))
        return self.total


class PairManager():
    def __init__(self, token_counter: Counter):
        self._pair_counter: dict[tuple[bytes, bytes],
                                 IndexManager] = defaultdict(IndexManager)
        self._pairs: list[tuple[tuple[bytes, ...], int]] = []
        self._next: list[list[int]] = []
        self._prev: list[list[int]] = []
        self._max_heap: MaxHeap = MaxHeap()
        self._initialize(token_counter)

    def _initialize(self, token_counter: Counter):
        self._pairs = [
            (tuple(bytes([byte]) for byte in token), count)
            for token, count in token_counter.items()
        ]
        self._next = [list(range(1, len(pair) + 1)) for pair, _ in self._pairs]
        self._prev = [list(range(-1, len(pair) - 1))
                      for pair, _ in self._pairs]
        self._initialize_heap()

    def _initialize_heap(self):
        for p_id, (tokens, _) in enumerate(self._pairs):
            for b_id, (byte1, byte2) in enumerate(zip(tokens[:-1], tokens[1:])):
                self._add((byte1, byte2), p_id, b_id)

    def _add(self, bytes_pair: tuple[bytes, bytes], p_id: int, b_id: int):
        total = self._pair_counter[bytes_pair].add(
            p_id, b_id, self._pairs[p_id][1])
        self._max_heap.push((total, bytes_pair))

    def _remove(self, bytes_pair: tuple[bytes, bytes], p_id: int, b_id: int):
        total = self._pair_counter[bytes_pair].remove(
            p_id, b_id, self._pairs[p_id][1])
        self._max_heap.push((total, bytes_pair))

    def update_index(self, bytes_pair: tuple[bytes, bytes]):
        cur_bytes_0, cur_bytes_1 = bytes_pair
        indices = self._pair_counter[bytes_pair].indices.copy()
        merged_bytes = cur_bytes_0 + cur_bytes_1
        last_p_id, last_b_id = -1, -1
        for p_id, b_id in sorted(indices):
            # Avoid overlap
            if p_id == last_p_id and self._next[p_id][last_b_id] > b_id:
                continue
            last_p_id, last_b_id = p_id, b_id
            line = self._pairs[p_id][0]
            # Example: a, (b, c), d
            # Remove: (a, b), (b, c), (c, d); Add: (a, bc), (bc, d)
            # 1. Remove (b, c)
            self._remove((cur_bytes_0, cur_bytes_1), p_id, b_id)
            # 2. Remove (c, d), Add(bc, d)
            next_1 = self._next[p_id][b_id]
            next_2 = self._next[p_id][next_1]
            if next_2 < len(self._next[p_id]):
                next_3 = self._next[p_id][next_2]
                next_bytes = b"".join(line[next_2: next_3])
                self._remove(
                    (cur_bytes_1, next_bytes), p_id, next_1)
                self._add(
                    (merged_bytes, next_bytes), p_id, b_id)
                self._prev[p_id][next_2] = b_id
            self._next[p_id][b_id] = next_2
            # 3. Remove (a, b), Add (a, bc)
            prev_1 = self._prev[p_id][b_id]
            if prev_1 >= 0:
                prev_bytes = b"".join(line[prev_1: b_id])
                self._remove(
                    (prev_bytes, cur_bytes_0), p_id, prev_1)
                self._add(
                    (prev_bytes, merged_bytes), p_id, prev_1)

    def get_max(self) -> tuple[bytes, bytes]:
        while not self._max_heap.is_empty():
            record_total, bytes_pair = self._max_heap.pop()
            if record_total == self._pair_counter[bytes_pair].total:
                return bytes_pair
        raise ValueError("No valid item found")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {
        **{i: token.encode("utf-8") for i, token in enumerate(special_tokens)},
        **{i + len(special_tokens): bytes([i]) for i in range(256)}
    }
    merges: list[tuple[bytes, bytes]] = []
    chunks = load_text_from_file(input_path)
    token_counter = Counter(pre_tokenize_train(chunks, special_tokens))
    pair_manager = PairManager(token_counter)

    while (len(vocab) < vocab_size):
        # Select the most frequent pairs with keyword: 1) frequency; 2) lexicographical greater.
        bytes_pair = pair_manager.get_max()
        # Merge and update indices
        pair_manager.update_index(bytes_pair)
        # Update Vocab
        next_id = len(vocab)
        vocab[next_id] = bytes_pair[0] + bytes_pair[1]
        merges.append(bytes_pair)
    return vocab, merges


def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], file_path: str):
    vocab_serializable = {str(k): v.hex()
                          for k, v in vocab.items()}  # hex encoding
    merges_serializable = [(pair[0].hex(), pair[1].hex()) for pair in merges]

    model_data = {
        'vocab': vocab_serializable,
        'merges': merges_serializable
    }

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)


def load_bpe_model(file_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    vocab = {int(k): bytes.fromhex(v) for k, v in model_data['vocab'].items()}
    merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1]))
              for pair in model_data['merges']]

    return vocab, merges

################################################################################
# Part II: Tokenizer
################################################################################


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.special_tokens_encoded: set[bytes] = set()
        self._handle_special_tokens()  # Append special token to vocab
        self.bacov: dict[bytes, int] = {
            token: id for id, token in vocab.items()}  # Inv of vocab
        self.merges_dict: dict[tuple[bytes, bytes], int] = {
            merge: idx for idx, merge in enumerate(merges)}

    def _handle_special_tokens(self):
        if self.special_tokens is not None:
            for token in self.special_tokens:
                token = token.encode("utf-8")
                self.special_tokens_encoded.add(token)
                if token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        raise ValueError("Not Implemented.")

    def _encode_token(self, token: bytes) -> list[int]:
        if not token:
            return []
        tokens: list[bytes] = [bytes([b]) for b in token]
        done = True
        while done:
            done = False
            apply_id, apply_order = -1, -1
            # Iterate and pick the earlist merge
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges_dict:
                    if apply_order == -1 or apply_order > self.merges_dict[pair]:
                        apply_id, apply_order = i, self.merges_dict[pair]
            # Apply the merge if available
            if apply_id != -1:
                tokens = tokens[:apply_id] + [tokens[apply_id] +
                                              tokens[apply_id + 1]] + tokens[apply_id + 2:]
                done = True
        return [self.bacov[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        text_bytes = pre_tokenize_infer((text,), self.special_tokens)
        result = []
        for token in text_bytes:
            if token in self.special_tokens_encoded:
                # Map special token to id directly
                result.append(self.bacov[token])
            else:
                result.extend(self._encode_token(token))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def encode_file2file(self,
                         input_path: str | os.PathLike,
                         output_path: str | os.PathLike,
                         chunk_size: int = (1 << 20)):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk = []

        token_generator = self.encode_iterable(load_text_from_file(input_path))
        temp_output_path = output_path.with_suffix('.bin.tmp')
        try:
            with open(temp_output_path, 'w') as f:
                for token_id in token_generator:
                    chunk.append(token_id)
                    if len(chunk) >= chunk_size:
                        # Batch write to file
                        np_array = np.array(chunk, dtype=np.int32)
                        np_array.tofile(f)
                        chunk = []

                # Write remaining tokens
                if chunk:
                    np_array = np.array(chunk, dtype=np.int32)
                    np_array.tofile(f)

            # Rename temporary file to final file
            temp_output_path.rename(output_path)

        except Exception as e:
            # If an error occurs, delete the temporary file
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise e

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors='replace')
