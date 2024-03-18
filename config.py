from dataclasses import dataclass


@dataclass
class TrainArgs:
    n_epochs: int = 10
    lr: float = 2e-5
    warmup_ratio: float = 0.01
    clip_grad_norm: float = 1.0


    contrastive_train_batch_size: int = 4
    contrastive_val_batch_size: int = 4
    embed_batch_size: int = 1024
    num_dataloader_workers: int = 0

    train_undersample_ratio: float = 0.01

    eval_steps = 100
    train_log_steps = 100
    save_steps = 100

    random_seed = 42
