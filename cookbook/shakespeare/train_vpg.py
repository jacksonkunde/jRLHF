def main():
    """
    Main function to set up and start policy gradient training using Jtransformer, JrewarderSimple,
    and a tiny dataset with CharTokenizer.
    """
    import os
    from datasets import Dataset

    from jtransformer.char_tokenizer import CharTokenizer
    from jtransformer.modules import Jtransformer
    from jRLHF.reward_model import JrewarderSimple
    from jRLHF.vanilla_policy_gradient import PolicyGradientTrainer
    from jRLHF.config import RLTrainingConfig

    # Initialize the tokenizer
    tokenizer_path = "base_jtransformers/char_tokenizer.json"
    tokenizer = CharTokenizer.load(tokenizer_path)

    # Create a tiny dataset of starter characters/words
    starter_texts = [
        "Hello",
        "Once",
        "The",
        "In",
        "On",
        "\n",
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "whom",
    ]

    tokenized_texts = tokenizer(
        starter_texts,
        padding="max_length",
        truncation=True,
        max_length=10,
        return_tensors="pt",
    )

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized_texts["input_ids"],
        }
    )

    train_data_path = "vpg_data/train_data"
    val_data_path = "vpg_data/val_data"

    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)

    dataset.save_to_disk(train_data_path)
    dataset.save_to_disk(val_data_path)

    # Initialize model
    model_path = "base_jtransformers/final"
    model = Jtransformer.load(model_path)

    # Initialize reward model
    token_to_reward = "h"
    token_id_to_reward = tokenizer(token_to_reward)["input_ids"][0]
    reward_model = JrewarderSimple(token_id=token_id_to_reward)

    training_cfg = RLTrainingConfig(
        n_epochs=5,
        batch_size=4,
        lr=1e-4,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        save_path="vpg_model_checkpoints",
        max_steps_per_epoch=10,
        save_freq=1,
        early_stopping_patience=3,
        wandb_project="PolicyGradientTraining",
        wandb_display_name="PolicyGradientRun",
        debug=True,
        max_new_tokens=20,
        temperature=1.0,
        scheduler_type=None,
    )

    trainer = PolicyGradientTrainer(
        cfg=training_cfg,
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
