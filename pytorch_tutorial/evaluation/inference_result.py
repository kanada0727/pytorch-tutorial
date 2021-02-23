from dataclasses import dataclass

import torch


@dataclass
class InferenceResult:
    images: torch.Tensor
    hidden_values: torch.Tensor
    labels: torch.Tensor

    def to_dict(self):
        return self.__dict__
