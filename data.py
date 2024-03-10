from torch.utils.data import Dataset
import random

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