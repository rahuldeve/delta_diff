import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DeltaModel(nn.Module):
    def __init__(self, base_transformer: AutoModel) -> None:
        super().__init__()
        self.embedder = base_transformer

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

        return {'loss': total_loss, 'id_loss': id_loss, 'inv_loss': inv_loss, 'delta_loss': delta_loss}