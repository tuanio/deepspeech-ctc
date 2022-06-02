import torch
import bpe
from typing import List
from abc import ABC, abstractmethod


class TextProcess(ABC):
    @abstractmethod
    def text2int(self, data):
        pass

    @abstractmethod
    def int2text(self, data):
        pass

    def decode(self, argmax: torch.Tensor):
        """
            decode greedy with collapsed repeat
        """
        decode = []
        for i, index in enumerate(arg_maxes):
            if index != self.blank_label:
                if i != 0 and index == arg_maxes[i - 1]:
                    continue
                decode.append(index.item())
        return self.int2text(decode)


class CharacterBased(TextProcess):
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"]
    blank_label = 0

    origin_list_vocab = {
        "en": aux_vocab + list("abcdefghijklmnopqrstuvwxyz"),
        "vi": aux_vocab
        + list(
            "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        ),
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang: str = "vi"):
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        return "".join([self.list_vocab[i] for i in s if i > 2])


class BPEBased(TextProcess):
    def __init__(self, **kwargs):
        self.encoder = bpe.Encoder(**kwargs)

    def fit(self, text_corpus: str = ""):
        return self.encoder.fit(text_corpus)

    def tokenize(self, text: str):
        return self.encoder.tokenize(text)

    def text2int(self, text: str):
        if isinstance(text, str):
            text = [text]
        return torch.Tensor(next(self.encoder.transform(text)))

    def int2text(self, idx: List[int]):
        return next(self.encoder.inverse_transform(idx))

    def load(self, in_path):
        self.encoder = self.encoder.load(in_path)