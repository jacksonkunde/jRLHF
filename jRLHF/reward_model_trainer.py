from typing import Optional, Dict, List

from datasets import Dataset, concatenate_datasets
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
        chunk_sizes: Optional[List[int]] = None,
        overlap_sizes: List[int] = [0],
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

        def pad(examples, max_length, pad_token_id):
            num_pad = max_length - len(examples["input_ids"])
            examples["input_ids"] = [pad_token_id] * num_pad + examples["input_ids"]
            return examples

        def compute_token_count_reward(examples, token_id):
            import numpy as np

            token_count_reward = np.sum(
                np.array(examples["input_ids"]) == token_id
            ).item()
            token_count_reward = min(token_count_reward, 100)
            return {"label": token_count_reward, "input_ids": examples["input_ids"]}

        if chunk_sizes:
            dataset_list = []
            for i, chunk_size in enumerate(chunk_sizes):
                # Define the overlap size, if only 1 is given, use that for all
                if len(overlap_sizes) == 1:
                    overlap_size = overlap_sizes[0]
                else:
                    overlap_size = overlap_sizes[i]
                tokenized_dataset = super().create_dataset(
                    tokenizer,
                    file_path,
                    hf_dataset_name,
                    tokenizer_kwargs,
                    chunk_size,
                    overlap_size,
                )
                dataset_list.append(tokenized_dataset)

            merged_dataset = concatenate_datasets(dataset_list)
            merged_dataset.map(
                pad,
                fn_kwargs={
                    "pad_token_id": tokenizer.pad_token_id,
                    "max_length": max(chunk_sizes),
                },
            )

            return merged_dataset.map(
                compute_token_count_reward, fn_kwargs={"token_id": token_id}
            )
        else:
            return (
                super()
                .create_dataset(
                    tokenizer,
                    file_path,
                    hf_dataset_name,
                    tokenizer_kwargs,
                    chunk_size=None,
                    overlap_size=0,
                )
                .map(compute_token_count_reward, fn_kwargs={"token_id": token_id})
            )
