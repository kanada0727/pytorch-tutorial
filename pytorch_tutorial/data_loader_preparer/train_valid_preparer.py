import torch
import torchvision
from sklearn.model_selection import train_test_split

from ..utils import join_with_root


class TrainValidPreparer:
    @classmethod
    def run(cls, config):
        total_dataset = torchvision.datasets.MNIST(
            join_with_root(config.data_loader.data_path),
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        train_dataset, val_dataset = cls._split_dataset(total_dataset)
        return (
            cls._init_dataloader(train_dataset, "train", config),
            cls._init_dataloader(val_dataset, "val", config),
        )

    @staticmethod
    def _split_dataset(total_dataset):
        train_indices, val_indices = train_test_split(
            list(range(len(total_dataset.targets))),
            test_size=0.1,
            stratify=total_dataset.targets,
        )
        train_dataset = torch.utils.data.Subset(total_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(total_dataset, val_indices)
        return train_dataset, val_dataset

    @staticmethod
    def _init_dataloader(dataset, type, config):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data_loader.batch_size[type],
            shuffle=True,
            num_workers=config.data_loader.num_workers,
        )
