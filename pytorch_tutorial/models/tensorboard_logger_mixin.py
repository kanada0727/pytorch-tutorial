class TensorboardLoggerMixin:
    def add_embedding(self, hidden_values, labels, images, phase_name):
        self.logger.experiment.add_embedding(
            hidden_values,
            metadata=labels,
            label_img=images,
            tag=f"{phase_name}_epoch{self.current_epoch}",
            global_step=self.global_step,
        )
