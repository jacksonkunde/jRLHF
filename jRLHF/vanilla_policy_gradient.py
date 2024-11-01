from typing import Dict, List, Tuple, Union
from tabulate import tabulate

import torch as th
import torch.nn.functional as F
from torch import nn, optim

import wandb

from jtransformer.char_tokenizer import CharTokenizer
from jtransformer.trainer import Jtrainer
from jRLHF.reward_model import Jrewarder
from jRLHF.config import RLTrainingConfig
from transformers import PreTrainedTokenizer


class PolicyGradientTrainer(Jtrainer):
    def __init__(
        self,
        cfg: RLTrainingConfig,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, CharTokenizer],
        reward_model: Jrewarder,
    ) -> None:
        """
        Initialize the PolicyGradientTrainer.

        Args:
            cfg (TrainingConfig): Training configuration.
            model (nn.Module): Language model to be trained.
            tokenizer (Union[PreTrainedTokenizer, CharTokenizer]): Tokenizer used for encoding text.
            reward_model (Jrewarder): Reward model to compute rewards for generated sequences.
        """
        super().__init__(cfg, model, tokenizer)
        self.reward_model = reward_model
        self.baseline = 0.0  # For baseline subtraction in policy gradient

    def _setup_loss(self) -> None:
        """
        No explicit loss function is needed for policy gradient training,
        as the loss is computed dynamically based on the rewards and log probabilities.
        """
        self.criterion = None

    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Set up the optimizer for policy gradient training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def val_metrics(self, logits: th.Tensor, targets: th.Tensor) -> Dict[str, float]:
        """
        Compute validation metrics.

        Args:
            logits (torch.Tensor): Model outputs (not used in policy gradient validation).
            targets (torch.Tensor): Ground truth targets (not used in policy gradient validation).

        Returns:
            Dict[str, float]: Dictionary of validation metrics.
        """
        # Val step will handle
        return {}

    def train_step(self, batch: Dict[str, th.Tensor]) -> float:
        """
        Perform a single training step using policy gradient.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data containing 'input_ids'.

        Returns:
            float: The computed loss value.
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)  # Shape: [batch_size, seq_len]

        # Generate sequences and obtain log probabilities
        generated_sequences, log_probs_list = self.generate_sequences(input_ids)

        # Compute rewards for the generated sequences
        rewards = self.compute_rewards(generated_sequences)  # Shape: [batch_size]

        # Optional: Update the baseline for variance reduction
        self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean().item()
        adjusted_rewards = rewards - self.baseline

        # Compute policy gradient loss
        loss = self.compute_policy_gradient_loss(log_probs_list, adjusted_rewards)

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        if not self.cfg.debug:
            wandb.log(
                {"train_loss": loss.item(), "train_reward": rewards.mean().item()},
                step=self.n_steps,
            )
        return loss.item()

    def generate_sequences(
        self, input_ids: th.Tensor
    ) -> Tuple[List[List[int]], List[th.Tensor]]:
        """
        Generate sequences by sampling from the model's output probabilities.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs. Shape: [batch_size, seq_len]

        Returns:
            Tuple[List[List[int]], List[torch.Tensor]]: Generated sequences and log probabilities.
        """
        batch_size = input_ids.size(0)
        generated_sequences = input_ids.tolist()  # Initialize with the input_ids
        log_probs_list = [
            [] for _ in range(batch_size)
        ]  # List to store log_probs per sequence

        max_new_tokens = self.cfg.max_new_tokens or 50
        eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        device = self.device

        is_finished = th.zeros(batch_size, dtype=th.bool, device=device)

        for _ in range(max_new_tokens):
            # Get logits from the model
            logits = self.model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]
            logits = logits[
                :, -1, :
            ]  # Take the logits of the last token. Shape: [batch_size, vocab_size]

            # Apply temperature
            temperature = self.cfg.temperature or 1.0
            logits = logits / temperature

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # Shape: [batch_size, vocab_size]

            # Create categorical distribution and sample
            distrib = th.distributions.Categorical(probs)
            actions = distrib.sample()  # Shape: [batch_size]

            log_probs = distrib.log_prob(actions)  # Shape: [batch_size]

            # Append the new tokens and log probabilities
            for i in range(batch_size):
                if not is_finished[i]:
                    action_token_id = actions[i].item()
                    generated_sequences[i].append(action_token_id)
                    log_probs_list[i].append(log_probs[i])

                    if action_token_id == eos_token_id:
                        is_finished[i] = True

            # Update input_ids for the next iteration
            actions = actions.unsqueeze(1)  # Shape: [batch_size, 1]
            input_ids = th.cat(
                [input_ids, actions], dim=1
            )  # Shape: [batch_size, seq_len + 1]

            # Break if all sequences are finished
            if is_finished.all():
                break

        # Convert log_probs_list to tensors
        for i in range(batch_size):
            if log_probs_list[i]:
                log_probs_list[i] = th.stack(
                    log_probs_list[i]
                )  # Shape: [seq_len_generated]
            else:
                # Handle the case where no tokens were generated
                log_probs_list[i] = th.tensor([], device=device)

        return generated_sequences, log_probs_list

    def compute_rewards(self, sequences: List[List[int]]) -> th.Tensor:
        """
        Compute rewards for each generated sequence using the reward model.

        Args:
            sequences (List[List[int]]): List of generated sequences.

        Returns:
            torch.Tensor: Tensor of rewards. Shape: [batch_size]
        """
        rewards = []
        for seq in sequences:
            reward = self.reward_model.get_reward(seq)
            rewards.append(reward)
        rewards = th.tensor(
            rewards, device=self.device, dtype=th.float32
        )  # Shape: [batch_size]
        return rewards

    def compute_policy_gradient_loss(
        self, log_probs_list: List[th.Tensor], rewards: th.Tensor
    ) -> th.Tensor:
        """
        Compute the policy gradient loss.

        Args:
            log_probs_list (List[torch.Tensor]): List of log probabilities for each sequence.
            rewards (torch.Tensor): Tensor of adjusted rewards. Shape: [batch_size]

        Returns:
            torch.Tensor: The computed loss (scalar).
        """
        losses = []
        for log_probs, reward in zip(log_probs_list, rewards):
            # Sum log probabilities over the sequence
            log_prob_sum = log_probs.sum()
            # Compute loss for this sequence
            losses.append(-log_prob_sum * reward)
        loss = th.stack(losses).mean()
        return loss

    def val_step(self, batch: Dict[str, th.Tensor]) -> dict:
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data containing 'input_ids'.

        Returns:
            dict: Validation metrics including average reward and number of datapoints.
        """
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)

        # Generate sequences
        with th.no_grad():
            generated_sequences, _ = self.generate_sequences(input_ids)

            # Compute rewards
            rewards = self.compute_rewards(generated_sequences)

        self.print_validation_examples(input_ids, generated_sequences, rewards)

        avg_reward = rewards.mean().item()

        return {"val_reward": avg_reward, "n_datapoints": input_ids.size(0)}

    def print_validation_examples(
        self,
        input_ids: th.Tensor,
        generated_sequences: List[List[int]],
        rewards: th.Tensor,
        num_examples: int = 3,
    ) -> None:
        """
        Helper function to print or log generated sequences and their rewards during validation.

        Args:
            input_ids (torch.Tensor): The input prompts. Shape: [batch_size, seq_len]
            generated_sequences (List[List[int]]): The generated sequences.
            rewards (torch.Tensor): The rewards for each sequence. Shape: [batch_size]
            num_examples (int, optional): Number of examples to print or log. Defaults to 3.
            log_to_wandb (bool, optional): Whether to log the examples to WandB. Defaults to False.
        """
        num_examples_to_print = min(num_examples, len(generated_sequences))
        examples = []
        for i in range(num_examples_to_print):
            # Decode the input prompt
            input_prompt = self.tokenizer.decode(input_ids[i])
            # Decode the generated sequence
            generated_text = self.tokenizer.decode(generated_sequences[i])
            # Get the reward
            reward = rewards[i].item()
            examples.append(
                {
                    "Prompt": input_prompt,
                    "Generated": generated_text,
                    "Reward": reward,
                }
            )

        if not self.cfg.debug:
            # Log examples to WandB
            table = wandb.Table(
                data=[list(ex.values()) for ex in examples],
                columns=list(examples[0].keys()),
            )
            wandb.log({"validation_examples": table}, step=self.n_steps)
        else:
            # Print examples
            print("\nValidation Step Examples:")
            headers = examples[0].keys()  # Extract column names from example keys
            rows = [ex.values() for ex in examples]  # Extract each row's values
            print(tabulate(rows, headers, tablefmt="grid"))

    def aggregate_metrics(
        self, val_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across the entire validation set using weighted averages.

        Args:
            val_metrics_list (List[Dict[str, float]]): List of validation metrics from each batch.

        Returns:
            Dict[str, float]: Aggregated validation metrics.
        """
        total_datapoints = sum(metrics["n_datapoints"] for metrics in val_metrics_list)
        aggregated = {}

        # Sum up each metric weighted by the number of datapoints in each batch
        for key in val_metrics_list[0].keys():
            if key == "n_datapoints":
                continue  # Skip this field in the aggregation
            weighted_sum = sum(
                metrics[key] * metrics["n_datapoints"] for metrics in val_metrics_list
            )
            aggregated[key] = (
                weighted_sum / total_datapoints
            )  # Compute weighted average

        return aggregated
