import torch

from .inference_result import InferenceResult


class InferenceResultAggregator:
    AGGREGATE_RULES = {
        "images": torch.vstack,
        "hidden_values": torch.vstack,
        "labels": torch.hstack,
    }

    @classmethod
    def run(cls, outputs):
        results = dict()
        for key, agg_func in cls.AGGREGATE_RULES.items():
            values = [out[key] for out in outputs]
            results[key] = agg_func(values)
        return InferenceResult(**results)
