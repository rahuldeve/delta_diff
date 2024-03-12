import random
from collections import defaultdict

import datasets as hds
import pandas as pd
from astartes import train_val_test_split
from torch.utils.data import Dataset

from utils import parallelize, standardize


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


def load_kasa_regression():
    df = pd.read_csv("./KasA_SMM_regression.csv")
    df = df[["SMILES", "Average Average Z Score"]]
    df = df.rename({"SMILES": "smiles", "Average Average Z Score": "score"}, axis=1)

    # df = standardize_df(df)

    splits = train_val_test_split(
        X=df["smiles"].to_numpy(),
        y=df["score"].to_numpy(),
        sampler="scaffold",
        random_state=42,
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