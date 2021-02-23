from .evaluation_result import EvaluationResult


class EvaluationResultWriter:
    def __init__(self, model, phase_name):
        self.model = model
        self.phase_name = phase_name

    def run(self, result: EvaluationResult):
        self.log(result, "accuracy")
        self.log(result, "loss")
        self.model.add_embedding(
            hidden_values=result.hidden_values,
            labels=result.labels.cpu().numpy(),
            images=result.images,
            phase_name=self.phase_name,
        )

    def log(self, result, key):
        self.model.log(f"{self.phase_name}_{key}", result[key])
