from .inference_result import InferenceResult


class Inferencer:
    @staticmethod
    def run(model, batch):
        inputs, targets = batch
        outputs = model.forward(inputs)

        return InferenceResult(
            images=inputs,
            hidden_values=outputs,
            labels=targets,
        ).to_dict()
