from typing import Optional, Dict

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

    def val_metrics(
        self, predictions: th.Tensor, targets: th.Tensor
    ) -> Dict[str, float]:
        val_loss = nn.functional.mse_loss(predictions, targets).item()
        return {"val_loss": val_loss}

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

        def compute_token_count_reward(examples, token_id):
            import numpy as np

            token_count_reward = np.sum(np.array(examples["input_ids"]) == token_id)
            token_count_reward = min(token_count_reward, 100)
            return {"label": token_count_reward, "input_ids": examples["input_ids"]}

        tokenized_dataset = super().create_dataset(
            tokenizer,
            file_path,
            hf_dataset_name,
            tokenizer_kwargs,
            chunk_size,
            overlap_size,
        )
        return tokenized_dataset.map(
            compute_token_count_reward, fn_kwargs={"token_id": token_id}
        )
