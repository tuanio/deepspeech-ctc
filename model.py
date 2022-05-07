import torch
from torchaudio.models import DeepSpeech
import pytorch_lightning as pl
from utils import TextProcess
import torch.nn.functional as F
import torchmetrics


class DeepSpeechModule(pl.LightningModule):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        dropout: float,
        n_class: int,
        lr: float,
        text_process: TextProcess,
        cfg_optim: dict,
    ):
        super().__init__()
        self.deepspeech = DeepSpeech(
            n_feature=n_feature, n_hidden=n_hidden, n_class=n_class, dropout=dropout
        )
        self.lr = lr
        self.text_process = text_process
        self.cal_wer = torchmetrics.WordErrorRate()
        self.cfg_optim = cfg_optim

    def forward(self, inputs):
        """predicting function"""
        if len(inputs.size()) == 3:
            # add batch
            inputs = inputs.unsqueeze(0)
        outputs = self.deepspeech(inputs)
        decode = outputs.argmax(dim=-1)
        predicts = [self.text_process.int2text(sent) for sent in decode]
        return predicts

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, **self.cfg_optim)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self.deepspeech(inputs)
        loss = F.ctc_loss(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        outputs = self.deepspeech(inputs)
        loss = F.ctc_loss(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )

        decode = outputs.argmax(dim=-1)

        predicts = [self.text_process.int2text(sent) for sent in decode]
        targets = [self.text_process.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("val_loss", loss)
        self.log("val_wer", wer)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self.deepspeech(inputs)

        loss = F.ctc_loss(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )

        decode = outputs.argmax(dim=-1)

        predicts = [self.text_process.int2text(sent) for sent in decode]
        targets = [self.text_process.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("test_loss", loss)
        self.log("test_wer", wer)

        return loss

    def log_output(self, predict, target, wer):
        print("=" * 50)
        print("Sample Predicts: ", predict)
        print("Sample Targets:", target)
        print("Mean WER:", wer)
        print("=" * 50)
