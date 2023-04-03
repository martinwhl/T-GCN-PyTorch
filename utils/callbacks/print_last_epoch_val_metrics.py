from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from rich import get_console
from rich.table import Table


def print_table_metrics(metrics):
    console = get_console()
    table = Table(header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)


class PrintLastEpochValMetrics(Callback):
    def __init__(self, as_table=True, to_logger=True):
        if as_table and not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichModelSummary` requires `rich` to be installed. Install it by running `pip install -U rich`."
            )
        self.as_table = as_table
        self.to_logger = to_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}
            if self.as_table:
                print_table_metrics(metrics)
            if self.to_logger:
                rank_zero_info(metrics)
