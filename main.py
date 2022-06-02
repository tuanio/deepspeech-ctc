import argparse
from utils import CharacterBased, BPEBased
from datasets import VivosDataset
from datamodule import VivosDataModule
from model import DeepSpeechModule
import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):
        if cfg.text.tokenizer == "character":
            text_process = CharacterBased(**cfg.text.character)
            n_class = len(text_process.list_vocab)
        elif cfg.text.tokenizer == "bpe":
            text_process = BPEBased(**cfg.text.bpe.params)
            n_class = cfg.text.bpe.params.vocab_size

        trainset = VivosDataset(**cfg.dataset, subset="train")

        if cfg.text.tokenizer == "bpe":
            if not cfg.text.bpe.is_train:
                print("Getting text corpus from train")
                text_corpus = [i[1] for i in trainset]
                print("Fitting text corpus to BPE...")
                text_process.fit(text_corpus)
                text_process.encoder.save(cfg.text.bpe.in_path)
            else:
                print("Load PBE from path...")
                text_process.load(cfg.text.bpe.in_path)

        testset = VivosDataset(**cfg.dataset, subset="test")

        dm = VivosDataModule(trainset, testset, text_process, **cfg.datamodule)

        model = DeepSpeechModule(
            n_class=n_class,
            text_process=text_process,
            cfg_optim=cfg.optimizer,
            **cfg.model
        )

        logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.logger)

        trainer = pl.Trainer(logger=logger, **cfg.trainer)

        if cfg.session.train:
            print("Training...")
            if cfg.ckpt.have_ckpt:
                trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
            else:
                trainer.fit(model, datamodule=dm)

        if cfg.session.validate:
            print("Validating...")
            trainer.validate(model, datamodule=dm)

        if cfg.session.test:
            print("Testing...")
            trainer.test(model, datamodule=dm)

    main()
