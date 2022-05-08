import torch
import ctcdecode


class TextProcess:

    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"]

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

    def __init__(self, lang):
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        return "".join([self.list_vocab[i] for i in s if i > 2])


class CTCDecoder:
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.96,
        beam_size: int = 100,
        kenlm_path: str = None,
        text_process: TextProcess = None
    ):  
        self.text_process = text_process
        labels = text_process.list_vocab
        blank_id = labels.index("<p>")

        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels,
            alpha=alpha,
            beta=beta,
            beam_width=beam_size,
            blank_id=blank_id,
            model_path=kenlm_path
        )
        print("finished loading beam search")

    def __call__(self, output: torch.Tensor) -> str:
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        tokens = beam_result[0][0]
        seq_len = out_seq_len[0][0]
        return self.text_process.int2text(tokens[:seq_len])