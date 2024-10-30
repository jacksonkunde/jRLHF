import os

from jRLHF.reward_model_trainer import RewardModelTrainer
from jRLHF.reward_model import Jrewarder
from jtransformer.config import TrainingConfig, TransformerConfig
from jtransformer.char_tokenizer import CharTokenizer


contex_window_size = 256
file_path = "input.txt"

tokenizer = CharTokenizer()

tokenizer_kwargs = {
    "max_length": contex_window_size,
    "padding": True,
    "truncation": True,
}
dataset = RewardModelTrainer.create_dataset(
    tokenizer=tokenizer,
    file_path=file_path,
    token_str="h",
    tokenizer_kwargs=tokenizer_kwargs,
    chunk_size=contex_window_size,
    overlap_size=contex_window_size // 2,
)
print(dataset)

tokenizer.save("char_tokenizer.json")

train_dataset = dataset[: int(len(dataset) * 0.9)]
val_dataset = dataset[int(len(dataset) * 0.9) :]

dataset.from_dict(train_dataset).save_to_disk("./train_dataset")
dataset.from_dict(val_dataset).save_to_disk("./val_dataset")

training_cfg = TrainingConfig(
    debug=False,
    batch_size=160,
    n_epochs=50,
    train_data_path=os.path.abspath("./train_dataset"),
    val_data_path=os.path.abspath("./val_dataset"),
    n_workers=4,
)

model_cfg = TransformerConfig(
    d_model=384,
    n_ctx=contex_window_size,
    d_mlp=4 * 384,
    n_heads=6,
    n_layers=6,
    d_vocab=len(tokenizer.vocab),
)

model = Jrewarder(model_cfg)

trainer = RewardModelTrainer(cfg=training_cfg, model=model, tokenizer=tokenizer)

trainer.train()
