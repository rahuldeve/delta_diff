import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from copy import deepcopy


class DeltaModel(nn.Module):
    def __init__(self, base_transformer: AutoModel) -> None:
        super().__init__()
        self.from_encoder = base_transformer
        self.to_encoder = deepcopy(base_transformer)

        hidden_size = base_transformer.config.hidden_size
        self.delta_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def embed_batch(self, batch):
        batch = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        embeddings_chunk = self.embedder(**batch).last_hidden_state
        embeddings_chunk = embeddings_chunk.sum(dim=-2)
        # embeddings_chunk = nn.functional.normalize(embeddings_chunk)
        return embeddings_chunk

    def get_delta(self, from_embedding, to_embedding):
        inp = torch.cat([from_embedding, to_embedding], axis=-1)
        return self.delta_head(inp)

    def identity_loss(self, from_embeddings, to_embeddings):
        diff_a = self.get_delta(from_embeddings, from_embeddings)
        diff_b = self.get_delta(to_embeddings, to_embeddings)

        # delta(a,a) ~ 0
        loss = diff_a.pow(2).sum().mean() + diff_b.pow(2).sum().mean()
        return loss

    def inversion_loss(self, from_embeddings, to_embeddings):
        diff_a = self.get_delta(from_embeddings, to_embeddings)
        diff_b = self.get_delta(to_embeddings, from_embeddings)

        # delta(a,b) ~ -delta(b, a)
        loss = F.mse_loss(diff_a, -diff_b)
        return loss

    def forward(self, batch):
        batch_from = batch["from"]
        batch_to = batch["to"]

        from_embeddings = self.embed_batch(batch_from)
        to_embeddings = self.embed_batch(batch_to)

        pred_delta = self.get_delta(from_embeddings, to_embeddings).squeeze()
        actual_delta = batch_from["score"] - batch_to["score"]

        # losses
        id_loss = self.identity_loss(from_embeddings, to_embeddings)
        inv_loss = self.inversion_loss(from_embeddings, to_embeddings)

        delta_loss = nn.functional.mse_loss(pred_delta, actual_delta)
        total_loss = delta_loss + id_loss + inv_loss
        return {
            "loss": total_loss,
            "id_loss": id_loss,
            "inv_loss": inv_loss,
            "delta_loss": delta_loss,
        }


class TripletDeltaModel(nn.Module):
    def __init__(self, base_transformer: AutoModel) -> None:
        super().__init__()
        self.embedder = base_transformer

        hidden_size = base_transformer.config.hidden_size
        self.delta_head = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

        self.cache = None

    def initialize_cache(self):
        self.cache = dict()

    def deallocate_cache(self):
        self.cache = None

    def embed_batch(self, batch):
        batch = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        embeddings_chunk = self.embedder(**batch).last_hidden_state[:, 0, :]
        return embeddings_chunk

    def get_delta(self, from_embedding, to_embedding):
        if self.cache is None:
            return self.calc_delta(from_embedding, to_embedding)

        from_ptr = from_embedding.data_ptr()
        to_ptr = to_embedding.data_ptr()
        key = (from_ptr, to_ptr)
        if key not in self.cache:
            self.cache[key] = self.calc_delta(from_embedding, to_embedding)
            return self.cache[key]
        else:
            return self.cache[key]

    def calc_delta(self, from_embedding, to_embedding):
        inp = torch.cat([from_embedding, to_embedding], axis=-1)
        return self.delta_head(inp).squeeze()

    def identity_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        anchor_loss = self.get_delta(anchor_embeddings, anchor_embeddings).pow(2).mean()
        hard_loss = self.get_delta(hard_embeddings, hard_embeddings).pow(2).mean()
        random_loss = self.get_delta(random_embeddings, random_embeddings).pow(2).mean()
        return (anchor_loss + hard_loss + random_loss) / 3

    def inversion_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        def helper(from_emb, to_emb):
            diff_a = self.get_delta(from_emb, to_emb)
            diff_b = self.get_delta(to_emb, from_emb)
            loss = F.mse_loss(diff_a, -diff_b)
            return loss

        anchor_hard_loss = helper(anchor_embeddings, hard_embeddings)
        anchor_random_loss = helper(anchor_embeddings, random_embeddings)
        hard_random_loss = helper(hard_embeddings, random_embeddings)
        loss = (anchor_hard_loss + anchor_random_loss + hard_random_loss) / 3
        return loss

    def triangular_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        diff_anchor_hard = self.get_delta(anchor_embeddings, hard_embeddings)
        diff_hard_random = self.get_delta(hard_embeddings, random_embeddings)
        diff_anchor_random = self.get_delta(anchor_embeddings, random_embeddings)
        loss = F.mse_loss(diff_anchor_hard + diff_hard_random, diff_anchor_random)
        return loss

    def delta_loss(
        self,
        anchor_embeddings,
        hard_embeddings,
        random_embeddings,
        anchor_scores,
        hard_scores,
        random_scores,
    ):
        pred_anchor_hard_delta = self.get_delta(anchor_embeddings, hard_embeddings)
        actual_anchor_hard_delta = anchor_scores - hard_scores
        anchor_hard_delta_loss = F.mse_loss(
            pred_anchor_hard_delta, actual_anchor_hard_delta
        )

        pred_anchor_random_delta = self.get_delta(anchor_embeddings, random_embeddings)
        actual_anchor_random_delta = anchor_scores - random_scores
        anchor_random_delta_loss = F.mse_loss(
            pred_anchor_random_delta, actual_anchor_random_delta
        )

        pred_hard_random_delta = self.get_delta(hard_embeddings, random_embeddings)
        actual_hard_random_delta = hard_scores - random_scores
        anchor_random_delta_loss = F.mse_loss(
            pred_hard_random_delta, actual_hard_random_delta
        )

        return (
            anchor_hard_delta_loss + anchor_random_delta_loss + anchor_random_delta_loss
        ) / 3

    def forward(self, batch):
        self.initialize_cache()

        batch_anchor = batch["anchor"]
        batch_hard = batch["hard"]
        batch_random = batch["random"]

        anchor_embeddings = self.embed_batch(batch_anchor)
        hard_embeddings = self.embed_batch(batch_hard)
        random_embeddings = self.embed_batch(batch_random)

        id_loss = self.identity_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        inv_loss = self.inversion_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        triangular_loss = self.triangular_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        delta_loss = self.delta_loss(
            anchor_embeddings,
            hard_embeddings,
            random_embeddings,
            batch_anchor["score"],
            batch_hard["score"],
            batch_random["score"],
        )
        total_loss = id_loss + inv_loss + triangular_loss + delta_loss

        self.deallocate_cache()

        return {
            "loss": total_loss,
            "id_loss": id_loss,
            "inv_loss": inv_loss,
            "triangular_loss": triangular_loss,
            "delta_loss": delta_loss,
        }


class MultiContrastiveModel(nn.Module):
    def __init__(self, base_transformer: AutoModel) -> None:
        super().__init__()
        self.embedder = base_transformer

        hidden_size = base_transformer.config.hidden_size
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

        self.cache = None

    def initialize_cache(self):
        self.cache = dict()

    def deallocate_cache(self):
        self.cache = None

    def embed_batch(self, batch):
        batch = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        embeddings_chunk = self.embedder(**batch).last_hidden_state[:, 0, :]
        return embeddings_chunk

    def get_delta(self, from_embedding, to_embedding):
        if self.cache is None:
            return self.calc_delta(from_embedding, to_embedding)

        from_ptr = from_embedding.data_ptr()
        to_ptr = to_embedding.data_ptr()
        key = (from_ptr, to_ptr)
        if key not in self.cache:
            self.cache[key] = self.calc_delta(from_embedding, to_embedding)
            return self.cache[key]
        else:
            return self.cache[key]

    def calc_delta(self, from_embedding, to_embedding):
        inp = torch.cat([from_embedding, to_embedding], axis=-1)
        return self.delta_head(inp).squeeze()

    def identity_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        anchor_loss = self.get_delta(anchor_embeddings, anchor_embeddings).pow(2).mean()
        hard_loss = self.get_delta(hard_embeddings, hard_embeddings).pow(2).mean()
        random_loss = self.get_delta(random_embeddings, random_embeddings).pow(2).mean()
        return (anchor_loss + hard_loss + random_loss) / 3

    def inversion_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        def helper(from_emb, to_emb):
            diff_a = self.get_delta(from_emb, to_emb)
            diff_b = self.get_delta(to_emb, from_emb)
            loss = F.mse_loss(diff_a, -diff_b)
            return loss

        anchor_hard_loss = helper(
            anchor_embeddings.unsqueeze(-2).expand_as(hard_embeddings), hard_embeddings
        )

        anchor_random_loss = helper(
            anchor_embeddings.unsqueeze(-2).expand_as(random_embeddings),
            random_embeddings,
        )

        hard_random_loss = helper(hard_embeddings, random_embeddings)

        loss = (anchor_hard_loss + anchor_random_loss + hard_random_loss) / 3
        return loss

    def triangular_loss(self, anchor_embeddings, hard_embeddings, random_embeddings):
        diff_anchor_hard = self.get_delta(
            anchor_embeddings.unsqueeze(-2).expand_as(hard_embeddings), hard_embeddings
        )

        diff_anchor_random = self.get_delta(
            anchor_embeddings.unsqueeze(-2).expand_as(random_embeddings),
            random_embeddings,
        )

        diff_hard_random = self.get_delta(hard_embeddings, random_embeddings)

        loss = F.mse_loss(diff_anchor_hard + diff_hard_random, diff_anchor_random)
        return loss

    def delta_loss(
        self,
        anchor_embeddings,
        hard_embeddings,
        random_embeddings,
        anchor_scores,
        hard_scores,
        random_scores,
    ):
        actual_anchor_hard_delta = anchor_scores.unsqueeze(-1) - hard_scores
        pred_anchor_hard_delta = self.get_delta(
            anchor_embeddings.unsqueeze(-2).expand_as(hard_embeddings), hard_embeddings
        )
        anchor_hard_delta_loss = F.mse_loss(
            pred_anchor_hard_delta, actual_anchor_hard_delta
        )

        actual_anchor_random_delta = anchor_scores.unsqueeze(-1) - random_scores
        pred_anchor_random_delta = self.get_delta(
            anchor_embeddings.unsqueeze(-2).expand_as(random_embeddings),
            random_embeddings,
        )
        anchor_random_delta_loss = F.mse_loss(
            pred_anchor_random_delta, actual_anchor_random_delta
        )

        actual_hard_random_delta = hard_scores - random_scores
        pred_hard_random_delta = self.get_delta(hard_embeddings, random_embeddings)
        anchor_random_delta_loss = F.mse_loss(
            pred_hard_random_delta, actual_hard_random_delta
        )

        return (
            anchor_hard_delta_loss + anchor_random_delta_loss + anchor_random_delta_loss
        ) / 3

    def forward(self, batch):
        # self.initialize_cache()

        batch_anchor = batch["anchor"]
        batch_hard = batch["hard"]
        batch_random = batch["random"]

        anchor_embeddings = self.embed_batch(batch_anchor)
        B = batch_hard["input_ids"].shape[0]
        n_hard = batch_hard["input_ids"].shape[1]
        n_rand = batch_random["input_ids"].shape[1]
        hard_embeddings = self.embed_batch(
            {k: v.view(B * n_hard, -1) for k, v in batch_hard.items()}
        ).view(B, n_hard, -1)

        random_embeddings = self.embed_batch(
            {k: v.view(B * n_rand, -1) for k, v in batch_random.items()}
        ).view(B, n_rand, -1)

        id_loss = self.identity_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        inv_loss = self.inversion_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        triangular_loss = self.triangular_loss(
            anchor_embeddings, hard_embeddings, random_embeddings
        )
        delta_loss = self.delta_loss(
            anchor_embeddings,
            hard_embeddings,
            random_embeddings,
            batch_anchor["score"],
            batch_hard["score"],
            batch_random["score"],
        )
        total_loss = id_loss + inv_loss + triangular_loss + delta_loss

        # self.deallocate_cache()

        return {
            "loss": total_loss,
            "id_loss": id_loss,
            "inv_loss": inv_loss,
            "triangular_loss": triangular_loss,
            "delta_loss": delta_loss,
        }
