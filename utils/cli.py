import lightning as L
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import utils.logging
from utils.callbacks import PrintLastEpochValMetrics, SaveLastEpochValResults


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # global arguments
        parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
        parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

        # argument linking
        parser.link_arguments("data.feat_max_val", "model.feat_max_val", apply_on="instantiate")
        parser.link_arguments("data.pre_len", "model.pre_len", apply_on="instantiate")
        parser.link_arguments("data.adj", "model.model.init_args.adj", apply_on="instantiate")
        parser.link_arguments("data.num_nodes", "model.model.init_args.num_nodes", apply_on="instantiate")
        parser.link_arguments("data.seq_len", "model.model.init_args.seq_len", apply_on="instantiate")

        # force callbacks
        parser.add_lightning_class_args(RichModelSummary, "callbacks.rich_model_summary")
        parser.add_lightning_class_args(RichProgressBar, "callbacks.rich_progress_bar")
        parser.add_lightning_class_args(PrintLastEpochValMetrics, "callbacks.print_last_epoch_val_metrics")
        parser.add_lightning_class_args(SaveLastEpochValResults, "callbacks.save_last_epoch_val_results")

    def before_fit(self):
        log_path = self.config.get("fit").get("log_path")
        if log_path is not None:
            utils.logging.output_logger_to_file(L.pytorch._logger, log_path)
        rank_zero_info(self.config)
