"""Context-aware sequential recommender after CARCA (Rashed, Elsayed & Schmidt-Thieme, RecSys 2022).

History encoder: each past session is the sum of its session-type embedding, a projection of its
attributes (distance, intensity, gap) and embeddings of its context (weekday, race calendar,
recovery). Causal masking is unnecessary because the history already ends before the target.
Self-attention layers encode the history.

Scorer: every candidate session type becomes a query built from its embedding, its static
attributes and the context of the day being planned. Cross-attention lets each candidate look
at the encoded history, and a linear head turns the result into one score per candidate.
All 66 candidates are scored at once and trained with a softmax cross-entropy.

`use_race_context=False` is the ablation: the race-calendar and recovery embeddings are
removed from both the history and the target context, everything else is identical.
"""

import torch
from torch import nn

from marathon.sequences import RACE_BUCKETS, RECOVERY_BUCKETS, SESSION_FEATURES, WEEKDAYS


class ContextCrossAttentionRecommender(nn.Module):
    def __init__(self, num_items, item_attributes, max_history, dim=64, heads=2, layers=2,
                 dropout=0.1, use_race_context=True):
        super().__init__()
        self.num_items = num_items
        self.use_race_context = use_race_context
        self.register_buffer("item_attributes", torch.as_tensor(item_attributes, dtype=torch.float32))
        self.item = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.session_features = nn.Linear(SESSION_FEATURES, dim)
        self.position = nn.Embedding(max_history, dim)
        self.weekday = nn.Embedding(WEEKDAYS, dim)
        self.race = nn.Embedding(RACE_BUCKETS, dim)
        self.recovery = nn.Embedding(RECOVERY_BUCKETS, dim)
        self.attributes = nn.Linear(self.item_attributes.shape[1], dim)
        self.target_weekday = nn.Embedding(WEEKDAYS, dim)
        self.target_race = nn.Embedding(RACE_BUCKETS, dim)
        self.target_recovery = nn.Embedding(RECOVERY_BUCKETS, dim)
        self.target_gap = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.history_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch, return_attention=False):
        """Scores of all candidates, shape [batch, num_items].

        With `return_attention=True`, also return the cross-attention weights of every candidate
        over the history, averaged over heads: shape [batch, num_items, history]. Each candidate's
        weights sum to 1 over the valid history positions and are 0 on padding. Call it in eval
        mode: in training mode the weights include attention dropout. The default call does not
        compute the weights, so training and evaluation are unchanged.
        """
        items, mask = batch["items"], batch["mask"]
        if items.ndim != 2 or mask.shape != items.shape:
            raise ValueError("items and mask must have shape [batch, history]")
        if not mask.any(dim=1).all():
            raise ValueError("Every example needs at least one history session")
        positions = torch.arange(items.shape[1], device=items.device)
        history = (self.item(items) + self.session_features(batch["features"]) + self.position(positions)[None]
                   + self.weekday(batch["weekday"]))
        context = self.target_weekday(batch["target_weekday"]) + self.target_gap(batch["target_gap"][:, None])
        if self.use_race_context:
            history = history + self.race(batch["race"]) + self.recovery(batch["recovery"])
            context = context + self.target_race(batch["target_race"]) + self.target_recovery(batch["target_recovery"])
        history = self.encoder(self.dropout(history), src_key_padding_mask=~mask)
        candidates = self.item.weight[1:] + self.attributes(self.item_attributes)
        queries = self.query_norm(candidates[None, :, :] + context[:, None, :])
        attended, weights = self.cross(queries, self.history_norm(history), self.history_norm(history),
                                       key_padding_mask=~mask, need_weights=return_attention,
                                       average_attn_weights=True)
        scores = self.head(queries + attended).squeeze(-1)
        return (scores, weights) if return_attention else scores
