from dataclasses import dataclass

import torch


@dataclass
class EvaluationResult:
    images: torch.Tensor
    hidden_values: torch.Tensor
    labels: torch.Tensor
    predictions: torch.tensor
    loss: torch.Tensor
    accuracy: float

    def to_numpy(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.cpu().numpy())
        return self

    def __getitem__(self, key):
        return getattr(self, key)
