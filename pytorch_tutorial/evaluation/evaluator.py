import torch
import torch.nn.functional as F

from .evaluation_result import EvaluationResult
from .inference_result import InferenceResult


class Evaluator:
    @classmethod
    def run(cls, result: InferenceResult) -> EvaluationResult:

        predictions = cls._predict_label(result.hidden_values)
        accuracy = cls._calc_accuracy(predictions, result.labels)
        loss = F.nll_loss(result.hidden_values, result.labels)

        return EvaluationResult(
            predictions=predictions,
            accuracy=accuracy,
            loss=loss,
            **result.to_dict(),
        )

    def _predict_label(hidden_values):
        return torch.max(hidden_values, 1).indices

    @staticmethod
    def _calc_accuracy(predictions, targets):
        acc = torch.sum(predictions == targets.data) / targets.shape[0]
        return acc
