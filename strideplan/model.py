"""Week 2 recurrent ranker and relative-effort head. Parameters start untrained."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class WorkoutRanker(nn.Module):
    """Template IDs are 1..num_templates; 0 pads variable-length histories.

    Session features and their order are defined by the caller's preprocessing.
    Fit feature scaling using training data only.
    Returns ranking logits and effort logits for every candidate template.
    """

    def __init__(self, num_templates, feature_size=5, hidden_size=32):
        super().__init__()
        if min(num_templates, feature_size, hidden_size) < 1:
            raise ValueError("Model dimensions must be positive")
        self.num_templates = num_templates
        self.feature_size = feature_size
        self.templates = nn.Embedding(num_templates + 1, hidden_size, padding_idx=0)
        self.encoder = nn.GRU(hidden_size + feature_size, hidden_size, batch_first=True)
        self.ranking_bias = nn.Parameter(torch.zeros(num_templates))
        self.effort_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                         nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, history_items, history_features, lengths):
        if history_items.ndim != 2 or history_items.shape[1] < 1:
            raise ValueError("history_items must have shape [batch, time] with at least one padded slot")
        batch, time = history_items.shape
        if batch < 1 or history_features.shape != (batch, time, self.feature_size):
            raise ValueError("history_features shape does not match history_items")
        if lengths.shape != (batch,) or lengths.dtype != torch.long or history_items.dtype != torch.long:
            raise ValueError("Item IDs and lengths must be int64 tensors with matching batch size")
        if not history_features.is_floating_point() or not torch.isfinite(history_features).all():
            raise ValueError("Session features must be finite floating-point values")
        if (lengths < 0).any() or (lengths > time).any():
            raise ValueError("Lengths must be between zero and the padded sequence length")
        valid = torch.arange(time, device=history_items.device)[None, :] < lengths.to(history_items.device)[:, None]
        if ((history_items[valid] < 1) | (history_items[valid] > self.num_templates)).any():
            raise ValueError("History template ID is outside the catalogue")
        if (history_items[~valid] != 0).any():
            raise ValueError("Unused history slots must have template ID zero")
        inputs = torch.cat((self.templates(history_items), history_features), dim=-1)
        packed = pack_padded_sequence(inputs, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        athlete = hidden[-1] * (lengths.to(hidden.device) > 0)[:, None]
        candidates = self.templates.weight[1:]
        ranking_logits = athlete @ candidates.T + self.ranking_bias
        pairs = torch.cat((athlete[:, None, :].expand(-1, self.num_templates, -1),
                           candidates[None, :, :].expand(batch, -1, -1)), dim=-1)
        effort_logits = self.effort_head(pairs).squeeze(-1)
        return ranking_logits, effort_logits
