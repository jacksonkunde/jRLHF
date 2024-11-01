from dataclasses import dataclass, field
from typing import Optional, Dict

from jtransformer.config import TrainingConfig


@dataclass
class RLTrainingConfig(TrainingConfig):
    max_new_tokens: int = 20
    temperature: float = 1.0
    scheduler_type: Optional[str] = None
    scheduler_kwargs: Dict = field(default_factory=dict)
    val_metric_name: Optional[str] = None
