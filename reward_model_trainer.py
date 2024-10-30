from jtransformer.trainer import Jtrainer
import torch as th
from torch import optim, nn


class RewardModelTrainer(Jtrainer):
    def _setup_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _setup_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def val_metric(self, predictions: th.Tensor, targets: th.Tensor) -> float:
        return nn.functional.mse_loss(predictions, targets).item()
