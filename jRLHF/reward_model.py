# Add reward model using the modules from jtransformer, but with linear projection
import os
import json

from jtransformer.modules import Embed, GPT2PositionalEmbed, TransformerBlock, LayerNorm
from jtransformer.config import TransformerConfig
import torch as th
import torch.nn as nn


class Jrewarder(nn.Module):
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

    def forward(self, input_ids) -> th.Tensor:
        res = self.embed(input_ids) + self.pos_embed(input_ids)
        for block in self.transformer_blocks:
            res = block(res)
        scalar_reward = self.out_proj(self.ln_last(res))
        return scalar_reward

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
    def load(cls, load_dir: str) -> "Jrewarder":
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
