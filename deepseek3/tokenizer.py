"""A tiny character-level tokenizer for Turkish names.

Every token is a single character. The vocabulary is built directly from the
names file, so it contains exactly the characters that appear in the data
(29 Turkish letters + the newline "\n", which we use as the start/end-of-name
marker).

Usage:
    tok = CharTokenizer.from_file("temiz_isimler.txt")
    ids = tok.encode("ali")        # -> [..]
    tok.decode(ids)                # -> "ali"
    tok.newline_id                 # id of "\n", the name separator / stop token
"""


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # char -> id
        self.itos = {i: ch for i, ch in enumerate(chars)}   # id -> char
        self.vocab_size = len(chars)
        # The newline both separates names and marks end-of-sequence (EOS).
        self.newline_id = self.stoi["\n"]
        self.eos_id = self.newline_id

    @classmethod
    def from_file(cls, path: str) -> "CharTokenizer":
        text = open(path, encoding="utf-8").read()
        if "\n" not in text:        # make sure the stop token always exists
            text += "\n"
        return cls(sorted(set(text)))

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
