from collections import defaultdict, deque
from functools import partial
from statistics import mean

import datasets as hds
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding

import wandb
from config import TrainArgs
from data import MultiContrastiveDS
from model import MultiContrastiveModel


def move_contrastive_batch_to_device(batch, device):
    for k1, subdict in batch.items():
        for k2, v in subdict.items():
            batch[k1][k2] = v.to(device)

    return batch


@torch.no_grad()
def embed_split(
    split_ds: hds.Dataset,
    model: MultiContrastiveModel,
    padding_collator: DataCollatorWithPadding,
    args: TrainArgs,
):
    embeddings = []
    device = next(model.parameters()).device
    for batch in DataLoader(
        split_ds,
        shuffle=False,
        batch_size=args.embed_batch_size,
        collate_fn=padding_collator,
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_embeddings = model.embed_batch(batch)
        embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.to(device)


@torch.no_grad()
def get_actual_diffs_for_split(split_ds: hds.Dataset):
    val_scores = torch.tensor(split_ds["score"])
    actual_diffs = val_scores.unsqueeze(-1) - val_scores.unsqueeze(0)
    return actual_diffs.cpu()


def get_chunk_idxs(N, chunk_size=64):
    st = 0
    en = chunk_size
    idxs = []
    while st <= N:
        idxs.append((st, en))
        st += chunk_size
        en += chunk_size

    return idxs


@torch.no_grad()
def get_predicted_diffs_for_split(
    split_ds: hds.Dataset,
    model: MultiContrastiveModel,
    padding_collator: DataCollatorWithPadding,
    args: TrainArgs,
):
    device = next(model.parameters()).device
    embeddings = embed_split(split_ds, model, padding_collator, args)

    N = embeddings.shape[0]
    pred_delta = torch.zeros((N, N))
    for row_idx in tqdm(range(N), desc="pairwise"):
        from_chunk = embeddings[[row_idx]].to(device)

        for chunk_st, chunk_en in get_chunk_idxs(N, 4096):
            to_chunk = embeddings[chunk_st:chunk_en].clone().to(device)

            pred_delta_row = model.get_delta(
                from_embedding=from_chunk.expand_as(to_chunk),
                to_embedding=to_chunk,
            )
            pred_delta[row_idx, chunk_st:chunk_en] = pred_delta_row.squeeze().cpu()

    return pred_delta.cpu()


def initialize_train_dataloader(train_split, model, padding_collator, args: TrainArgs):
    train_ds = MultiContrastiveDS(
        train_split,
        get_actual_diffs_for_split(train_split),
        get_predicted_diffs_for_split(train_split, model, padding_collator, args),
        padding_collator,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.contrastive_train_batch_size,
        shuffle=True,
        collate_fn=train_ds.get_contrastive_collator(),
        drop_last=True,
        num_workers=args.num_dataloader_workers,
    )

    return train_dl


class LossAccumulator:
    def __init__(self, num_steps) -> None:
        self.acc_map = defaultdict(partial(deque, maxlen=num_steps))

    def append(self, loss_map):
        for k, v in loss_map.items():
            self.acc_map[k].append(v.item())

    def summarize(self):
        return {k: round(mean(v), 4) for k, v in self.acc_map.items()}


class MultiContrastiveTrainer:
    def __init__(
        self,
        model: MultiContrastiveModel,
        optimizer,
        scheduler,
        train_dl: DataLoader,
        val_ds: hds.Dataset,
        args: TrainArgs,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

        self.train_dl = train_dl
        self.val_ds = val_ds

        self.global_step = 0
        self.global_epoch = 0
        self.train_loss_accumulator = LossAccumulator(args.train_log_steps)

        wandb.watch(self.model, log="gradients", log_freq=100)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        self.model.train()
        device = next(self.model.parameters()).device

        batch = move_contrastive_batch_to_device(batch, device)
        losses = self.model(batch)
        losses["loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.clip_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        return losses

    def update_hard_negatives(self):
        train_ds: MultiContrastiveDS = self.train_dl.dataset
        padding_collator = train_ds.get_base_collator()
        updated_predicted_deltas = get_predicted_diffs_for_split(
            split_ds=train_ds.dataset,
            model=self.model,
            padding_collator=padding_collator,
            args=self.args,
        )

        train_ds.predicted = updated_predicted_deltas
        self.train_dl.dataset = train_ds

    def validate(self):
        self.model.eval()
        actual_diffs = get_actual_diffs_for_split(self.val_ds)
        pred_diffs = get_predicted_diffs_for_split(
            self.val_ds,
            self.model,
            self.train_dl.dataset.get_base_collator(),
            self.args,
        )

        errors = pred_diffs - actual_diffs
        wandb.log(
            {
                "val/epoch": self.global_epoch,
                "val/delta_mse": errors.pow(2).mean(),
                "val/delta_r2": r2_score(
                    actual_diffs.cpu().numpy(), pred_diffs.cpu().numpy()
                ),
                "val/delta_mean_worst_error": errors.abs().max(dim=-1).values.mean(),
                "val/delta_worst_error": wandb.Histogram(errors.abs()),
            }
        )

    def step_triggers(self):
        if self.global_step % self.args.eval_steps == 0:
            self.validate()

        if self.global_step % self.args.train_log_steps == 0:
            train_loss_summary = self.train_loss_accumulator.summarize()
            train_loss_summary = {
                f"train/{k}": v for k, v in train_loss_summary.items()
            }
            train_loss_summary["train/epoch"] = self.global_epoch
            wandb.log(train_loss_summary, step=self.global_step)

    def loop_epoch(self):
        for batch in tqdm(
            self.train_dl, total=len(self.train_dl), desc="Train: ", leave=True
        ):
            losses = self.train_step(batch)
            self.train_loss_accumulator.append(losses)

            self.global_step += 1
            self.step_triggers()

        self.global_epoch += 1

    def train(self):
        self.validate()
        for e in tqdm(range(self.args.n_epochs), desc="Epoch: "):
            self.loop_epoch()

        self.validate()
