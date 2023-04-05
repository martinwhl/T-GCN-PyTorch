import numpy as np
import lightning as L


class SaveLastEpochValResults(L.Callback):
    def __init__(self, save_path=None):
        super(SaveLastEpochValResults, self).__init__()
        self.ground_truths = []
        self.predictions = []
        self.save_path = save_path

    def on_validation_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.current_epoch == trainer.max_epochs - 1:
            predictions, targets = outputs
            self.ground_truths.append(targets.detach().cpu().numpy())
            self.predictions.append(predictions.detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            ground_truths = np.concatenate(self.ground_truths, axis=0)
            predictions = np.concatenate(self.predictions, axis=0)
            if self.save_path is None:
                self.save_path = pl_module.logger.experiment.log_dir + "/test_results.npz"
            np.savez(self.save_path, outputs=predictions, targets=ground_truths)
