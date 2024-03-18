import random
from collections import defaultdict
from functools import partial
from itertools import chain

import datasets as hds
import pandas as pd
from astartes import train_val_test_split
from torch.utils.data import Dataset

from utils import parallelize, standardize
from config import TrainArgs


@parallelize(8)
def standardize_df(df):
    df["inchi"] = df["smiles"].map(standardize)
    return df


def tokenize(entry, tokenizer):
    entry = dict(entry)
    smiles = entry.pop("smiles")
    encoded = tokenizer(
        smiles,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )

    return encoded


def load_kasa_regression(args: TrainArgs):
    df = pd.read_csv("./KasA_SMM_regression.csv")
    df = df[["SMILES", "Average Average Z Score"]]
    df = df.rename({"SMILES": "smiles", "Average Average Z Score": "score"}, axis=1)

    df = standardize_df(df)

    splits = train_val_test_split(
        X=df["smiles"].to_numpy(),
        y=df["score"].to_numpy(),
        sampler="scaffold",
        random_state=args.random_seed,
        return_indices=True,
    )

    train_ids, val_ids, test_ids = splits[-3], splits[-2], splits[-1]
    df_train = df.iloc[train_ids].reset_index(drop=True)
    df_val = df.iloc[val_ids].reset_index(drop=True)
    df_test = df.iloc[test_ids].reset_index(drop=True)

    return hds.DatasetDict(
        {
            "train": hds.Dataset.from_pandas(df_train),
            "val": hds.Dataset.from_pandas(df_val),
            "test": hds.Dataset.from_pandas(df_test),
        }
    )


def pair_collate(batch, base_collator):
    batch_from = [item["from"] for item in batch]
    batch_to = [item["to"] for item in batch]

    batch_from = base_collator(batch_from)
    batch_to = base_collator(batch_to)

    return {"from": batch_from, "to": batch_to}


def batch_of_dict_collate(batch, base_collator):
    # convert from list of dicts to dict of lists
    dict_of_lists = defaultdict(list)
    for entry in batch:
        for k, v in entry.items():
            dict_of_lists[k].append(v)

    collated_batch = {k: base_collator(v) for k, v in dict_of_lists.items()}
    return collated_batch


# Try contrastive
class PairDS(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        from_entry = self.ds[index]

        to_rand_idx = random.randint(0, len(self.ds) - 1)
        to_rand_entry = self.ds[to_rand_idx]
        return {"from": from_entry, "to": to_rand_entry}


class ContrastivePairDS(Dataset):
    def __init__(self, ds, actual, predicted):
        self.ds = ds
        self.actual = actual
        self.predicted = predicted

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        from_entry = self.ds[index]

        hard_idx = (self.predicted[index] - self.actual[index]).abs().argmax(dim=-1)
        hard_idx = hard_idx.item()
        to_hard_entry = self.ds[hard_idx]
        return {"from": from_entry, "to": to_hard_entry}


class ContrastiveTripletDS(Dataset):
    def __init__(self, ds, actual, predicted):
        self.ds = ds
        self.actual = actual
        self.predicted = predicted

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        anch_entry = self.ds[index]

        hard_idx = (self.predicted[index] - self.actual[index]).abs().argmax(dim=-1)
        hard_idx = hard_idx.item()
        hard_entry = self.ds[hard_idx]

        rand_idx = random.randint(0, len(self.ds) - 1)
        rand_entry = self.ds[rand_idx]

        return {"anchor": anch_entry, "hard": hard_entry, "random": rand_entry}


def list_of_dicts_to_dict_of_list(batch):
    dict_of_lists = defaultdict(list)
    for entry in batch:
        for k, v in entry.items():
            dict_of_lists[k].append(v)

    return dict(dict_of_lists)


def multiple_contrastive_collate(batch, base_collator):
    # convert from list of dicts to dict of lists
    batch_collected = list_of_dicts_to_dict_of_list(batch)
    batch_collected = {
        k: list_of_dicts_to_dict_of_list(v) for k, v in batch_collected.items()
    }

    batch_collected["anchor"] = base_collator(batch_collected["anchor"])
    batch_collected["hard"] = base_collator(
        {k: list(chain.from_iterable(v)) for k, v in batch_collected["hard"].items()}
    )
    batch_collected["random"] = base_collator(
        {k: list(chain.from_iterable(v)) for k, v in batch_collected["random"].items()}
    )
    batch_collated = {k: base_collator(v) for k, v in batch_collected.items()}

    B = len(batch)
    n_hard = len(batch_collected["hard"]["score"]) // B
    n_random = len(batch_collected["random"]["score"]) // B
    batch_collated["hard"] = {
        k: v.reshape(B, n_hard, -1).squeeze() for k, v in batch_collated["hard"].items()
    }
    batch_collated["random"] = {
        k: v.reshape(B, n_random, -1).squeeze()
        for k, v in batch_collated["random"].items()
    }
    return batch_collated


class MultiContrastiveDS(Dataset):
    def __init__(self, ds, actual, predicted, base_collator):
        self.ds = ds
        self.actual = actual
        self.predicted = predicted
        self.n_candidates = 64
        self.n_hard = int(0.5 * self.n_candidates)
        self.n_random = self.n_candidates - self.n_hard
        self.base_collator = base_collator

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        anch_entry = self.ds[index]

        pred_actual_diffs = (self.predicted[index] - self.actual[index]).abs()
        topk_hard_idxs = pred_actual_diffs.topk(k=self.n_hard).indices
        hard_entries = self.ds[topk_hard_idxs]

        rand_idxs = [random.randint(0, len(self.ds) - 1) for _ in range(self.n_random)]
        rand_entries = self.ds[rand_idxs]

        return {"anchor": anch_entry, "hard": hard_entries, "random": rand_entries}

    def get_base_collator(self):
        return self.base_collator

    def get_contrastive_collator(self):
        return partial(multiple_contrastive_collate, base_collator=self.base_collator)
