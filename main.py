import lightning as L
import utils.logging
from tasks import SupervisedForecastTask
from utils.data import SpatioTemporalCSVDataModule
from utils.cli import CustomLightningCLI


def main():
    utils.logging.format_logger(L.pytorch._logger)
    cli = CustomLightningCLI(SupervisedForecastTask, SpatioTemporalCSVDataModule)  # noqa: F841


if __name__ == "__main__":
    main()
