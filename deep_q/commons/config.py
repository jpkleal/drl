from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DeepQConfig:
    # Torch Parameters
    device: torch.device

    # Net Parameters
    n_inputs: int
    n_outputs: int
    net_kwargs: Optional[dict]

    # Memory Parameters
    rm_size: int
    batch_size: int

    # Deep Q Hyperparameters
    lr: float
    gamma: float
    tau: float

    # Epsilon
    epsilon_kwargs: Optional[dict]

    # Other
    epochs: int
    save_rate: int
