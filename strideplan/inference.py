"""Serve next-workout rankings from a saved bundle with the training preprocessing."""

import csv
from collections import defaultdict

import numpy as np
import torch

from strideplan.features import add_raw_features, describe_template, parse_session, scale_features, template_id


def encode_history(records, catalogue, scaler, max_speed, max_history):
    """Return the templates and scaled features of the latest usable sessions, oldest first.

    Gaps are computed on the full cleaned history before truncation, as in training.
    """
    rows = [row for row, _ in (parse_session(record, max_speed) for record in records) if row is not None]
    add_raw_features(rows)
    rows = rows[-max_history:]
    templates = [template_id(row, catalogue) for row in rows]
    features = [scale_features(row["raw_features"], scaler) for row in rows]
    return templates, features


def load_histories(path):
    """Group processed-CSV sessions by user. Serving may use every split: it predicts after the dataset ends."""
    histories = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            histories[record["user_id"]].append(record)
    return dict(histories)


class Recommender:
    def __init__(self, bundle):
        self.bundle = bundle
        self.catalogue = bundle.catalogue
        self.templates = {template: describe_template(template, self.catalogue)
                          for template in range(1, self.catalogue["num_templates"] + 1)}

    def rank(self, records):
        """Rank every template. Users with no usable session receive global popularity."""
        templates, features = encode_history(records, self.catalogue, self.bundle.scaler,
                                             self.bundle.max_speed, self.bundle.max_history)
        if not templates:
            return {"strategy": "global_popularity", "history_sessions_used": 0,
                    "ranking": [(template, None) for template in self.bundle.popularity]}
        items = torch.tensor([templates], dtype=torch.long)
        values = torch.from_numpy(np.stack(features))[None]
        with torch.no_grad():
            logits, _ = self.bundle.model(items, values, torch.tensor([len(templates)]))
        probabilities = torch.softmax(logits[0], dim=0)
        # Stable sort resolves ties by ascending template ID, as in offline evaluation.
        order = torch.argsort(logits[0], descending=True, stable=True) + 1
        return {"strategy": "gru", "history_sessions_used": len(templates),
                "ranking": [(int(template), float(probabilities[template - 1])) for template in order]}

    def recommend(self, records, max_duration_minutes, top_k):
        """Keep categories whose shortest duration fits the budget, cap their duration, then cut to top_k."""
        result = self.rank(records)
        recommendations = []
        for template, score in result["ranking"]:
            description = self.templates[template]
            if description["duration_min_minutes"] > max_duration_minutes:
                continue
            recommendations.append({**description, "score": score,
                                    "duration_max_minutes": min(description["duration_max_minutes"],
                                                                max_duration_minutes)})
            if len(recommendations) == top_k:
                break
        return {"strategy": result["strategy"], "history_sessions_used": result["history_sessions_used"],
                "recommendations": recommendations}
