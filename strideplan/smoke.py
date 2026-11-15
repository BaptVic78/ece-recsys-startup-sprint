"""Run the Week 2 model and evaluation checks on explicitly synthetic data."""

import json

import torch
from torch.nn import functional as F

from strideplan.evaluation import evaluate, most_popular
from strideplan.model import WorkoutRanker


def main():
    torch.manual_seed(42)
    model = WorkoutRanker(num_templates=3)
    items = torch.tensor([[1, 2, 3], [2, 0, 0], [0, 0, 0]])
    features = torch.tensor([[[0.3, 0.5, 0.1, 0.2, 0.6], [0.5, 0.6, 0.2, 0.3, 0.7], [0.8, 0.5, 0.3, 0.4, 0.8]],
                             [[0.5, 0.5, 0.2, 0.2, 0.6], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
                             [[0., 0., 0., 0., 0.]] * 3])
    ranking_logits, effort_logits = model(items, features, torch.tensor([3, 1, 0]))
    loss = F.cross_entropy(ranking_logits, torch.tensor([1, 2, 0]))
    loss += F.binary_cross_entropy_with_logits(effort_logits, torch.zeros_like(effort_logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    print(json.dumps({"data": "synthetic", "trained": False,
                      "ranking_shape": list(ranking_logits.shape), "effort_shape": list(effort_logits.shape),
                      "forward_and_backward": "passed",
                      "mock_metrics": evaluate({"a": [1, 2, 3], "b": [3, 2, 1]}, {"a": {1}, "b": {2}}),
                      "mock_popularity_order": most_popular([[1, 1, 2], [2, 3, 1]], [1, 2, 3])}, indent=2))


if __name__ == "__main__":
    main()
