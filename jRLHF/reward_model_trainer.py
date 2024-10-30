from typing import Optional

from datasets import Dataset
from jtransformer.trainer import Jtrainer

import torch as th
from torch import optim, nn
from transformers import PreTrainedTokenizer


class RewardModelTrainer(Jtrainer):
    def _setup_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _setup_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def val_metric(self, predictions: th.Tensor, targets: th.Tensor) -> float:
        return nn.functional.mse_loss(predictions, targets).item()

    @classmethod
    def create_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        token_id: Optional[int] = None,
        token_str: Optional[str] = None,
        file_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        tokenizer_kwargs: dict = {},
        chunk_size: Optional[int] = None,
        overlap_size: int = 0,
    ) -> Dataset:
        if not token_id:
            assert (
                token_str is not None
            ), "You must pass either a token_id (int) or a token_str (the str representation of the token)."
            token_ids = tokenizer(token_str)["input_ids"]
            if len(token_ids) == 2:
                # Assume there is a begginning of string token
                token_id = token_ids[1]
            elif len(token_ids) > 2:
                raise Exception("The token_str you passed is more than one token.")
            else:
                token_id = token_ids[0]

        def compute_percentage_reward(examples, token_id):
            import numpy as np

            percentage_reward = np.mean(np.array(examples["input_ids"]) == token_id)
            return {"percentage_reward": percentage_reward}

        tokenized_dataset = super().create_dataset(
            tokenizer,
            file_path,
            hf_dataset_name,
            tokenizer_kwargs,
            chunk_size,
            overlap_size,
        )
        return tokenized_dataset.map(compute_percentage_reward, token_id)