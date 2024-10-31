# Add reward model using the modules from jtransformer, but with linear projection
import os
import json
from typing import Optional, List, Union

from jtransformer.modules import Embed, GPT2PositionalEmbed, TransformerBlock, LayerNorm
from jtransformer.config import TransformerConfig
import torch as th
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod


class Jrewarder(ABC):
    @abstractmethod
    def get_reward(self, input_ids: Union[List[int], th.Tensor], **kwargs) -> float:
        pass


class JrewarderSimple(Jrewarder):
    """Simple rewarder that will count the occurances of a give token in the input sequence."""

    def __init__(self, token_id) -> None:
        super().__init__()
        self.token_id = token_id

    def get_reward(self, input_ids: List[int] | th.Tensor, **kwargs) -> float:
        if isinstance(input_ids, th.Tensor):
            input_ids = input_ids.tolist()
        token_count_reward = np.sum(np.array(input_ids) == self.token_id).item()
        token_count_reward = min(token_count_reward, 100)
        return token_count_reward


class JrewarderLM(nn.Module, Jrewarder):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = GPT2PositionalEmbed(cfg)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_last = LayerNorm(cfg)
        self.out_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self, input_ids, attention_mask: Optional[th.Tensor] = None
    ) -> th.Tensor:
        res = self.embed(input_ids) + self.pos_embed(input_ids)
        for block in self.transformer_blocks:
            res = block(res, attention_mask=attention_mask)
        scalar_reward = self.out_proj(self.ln_last(res))
        return scalar_reward[:, -1]

    def get_reward(self, input_ids, **kwargs):
        return self.forward(
            input_ids, attention_mask=kwargs.get("attention_mask", None)
        ).item

    def save(self, save_dir: str) -> None:
        """
        Save the model's state_dict and configuration to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save the model's state dict
        model_path = os.path.join(save_dir, "model.pth")
        th.save(self.state_dict(), model_path)

        # Save the configuration as JSON
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg.__dict__, f, indent=4)

        print(f"Model and config saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str) -> "JrewarderLM":
        """
        Load the model's state_dict and configuration from the given directory.
        """
        # Load the configuration from JSON
        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # Recreate the config object
        cfg = TransformerConfig(**cfg_dict)

        # Initialize the model with the loaded config
        model = cls(cfg)

        # Load the state dict
        model_path = os.path.join(load_dir, "model.pth")
        model.load_state_dict(th.load(model_path, map_location="cpu"))

        print(f"Model loaded from {load_dir}")
        return model
