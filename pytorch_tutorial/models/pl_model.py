import hydra
import pytorch_lightning as pl
import torch.nn.functional as F

from ..evaluation.evaluation_result_writer import EvaluationResultWriter
from ..evaluation.evaluator import Evaluator
from ..evaluation.inference_result_aggregator import InferenceResultAggregator
from ..evaluation.inferencer import Inferencer
from .tensorboard_logger_mixin import TensorboardLoggerMixin


class PlModel(pl.LightningModule, TensorboardLoggerMixin):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.hparams = dict(config.hparams)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self.model(data)
        loss = F.nll_loss(out, target)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, "val")

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "test")

    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "val")

    def _evaluation_step(self, batch, batch_idx, phase_name):
        return Inferencer.run(self, batch)

    def _evaluation_epoch_end(self, outputs, phase_name):
        outputs = InferenceResultAggregator.run(outputs)
        result = Evaluator.run(outputs)
        EvaluationResultWriter(self, phase_name).run(result)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.model.parameters())
        return optimizer
