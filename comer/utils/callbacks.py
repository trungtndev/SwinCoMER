import pytorch_lightning as pl
from setproctitle import setproctitle


class ProgressProcTitle(pl.Callback):
    def __init__(self, base_name: str, monitor: str = "val_loss"):
        super().__init__()
        self.base_name = base_name
        self.monitor = monitor
        self.metric = 0.0
        self.title_template = "{base_name} | epoch {current_epoch}/{max_epochs} | {metric_value:.4f}"

    def on_train_epoch_start(self, trainer, pl_module):
        self._update_title(trainer)

    def on_validation_end(self, trainer, pl_module):
        self._update_title(trainer)

    def _update_title(self, trainer):
        if not trainer.is_global_zero:
            return

        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        metric_value = trainer.callback_metrics.get(self.monitor)

        if metric_value is not None:
            try:
                metric_value = float(metric_value)
                self.metric = metric_value
            except Exception:
                metric_value = self.metric
        else:
            metric_value = self.metric

        title = self.title_template.format(
            base_name=self.base_name,
            current_epoch=current_epoch,
            max_epochs=max_epochs,
            metric_value=metric_value,
        )

        setproctitle(title)
