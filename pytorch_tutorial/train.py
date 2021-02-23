import hydra
from hydra.utils import instantiate

from .data_loader_preparer.train_valid_preparer import TrainValidPreparer
from .models.pl_model import PlModel


@hydra.main(config_path="./", config_name="config.yml")
def main(config):
    train_loader, val_loader = TrainValidPreparer.run(config)
    model = instantiate(config.hparams.model)
    pl_model = PlModel(model, config)
    trainer = instantiate(config.trainer)

    trainer.fit(pl_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
