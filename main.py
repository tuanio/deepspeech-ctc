import argparse
from utils import TextProcess
from datasets import VivosDataset
from datamodule import VivosDataModule
from model import DeepSpeechModule
import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):
        text_process = TextProcess(**cfg.text_process)

        trainset = VivosDataset(**cfg.dataset, subset="train")
        testset = VivosDataset(**cfg.dataset, subset="test")

        dm = VivosDataModule(trainset, testset, text_process, **cfg.datamodule)

        n_class = len(text_process.list_vocab)
        model = DeepSpeechModule(
            n_class=n_class, text_process=text_process, **cfg.model
        )

        logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.logger)

        trainer = pl.Trainer(logger=logger, **cfg.trainer)
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

    main()
