"""Short feedback loop: adjust a model ranking with a user's recent feedback, without retraining.

Every adjustment is additive in log-utility and is explained in the item's reasons.
"""

import math
from dataclasses import dataclass


EFFORT_DIRECTION = {"too_easy": 1.0, "ok": 0.0, "too_hard": -1.0}


@dataclass(frozen=True)
class Policy:
    rejection_penalty: float = 1.5
    rejection_half_life_days: float = 7.0
    effort_shift: float = 1.0
    effort_half_life_days: float = 14.0
    effort_window: int = 5
    learned_effort_weight: float = 1.0
    epsilon: float = 0.1


NO_FEEDBACK = Policy(rejection_penalty=0.0, effort_shift=0.0, learned_effort_weight=0.0, epsilon=0.0)


def decay(age_seconds, half_life_days):
    return 0.5 ** (max(0.0, age_seconds) / (half_life_days * 86400))


def intensity(template, catalogue):
    """Position of a category between the lightest (-1) and the hardest (+1) duration and speed bands."""
    speed_count = len(catalogue["speed_edges_mps"]) + 1
    duration_count = len(catalogue["duration_edges_seconds"]) + 1
    duration_band, speed_band = divmod(template - 1, speed_count)
    parts = [band / (count - 1) for band, count in ((duration_band, duration_count), (speed_band, speed_count))
             if count > 1]
    return 2 * sum(parts) / len(parts) - 1 if parts else 0.0


def effort_bias(feedback, now, policy):
    """Positive after 'too easy', negative after 'too hard'. Fades with age and is bounded by [-1, 1]."""
    rated = [event for event in feedback if event["action"] == "completed" and event["effort"]]
    rated = rated[-policy.effort_window:]
    if not rated:
        return 0.0
    total = sum(EFFORT_DIRECTION[event["effort"]] * decay(now - event["created_at"], policy.effort_half_life_days)
                for event in rated)
    return total / len(rated)


def rejection_weights(feedback, now, policy):
    weights = {}
    for event in feedback:
        if event["action"] == "rejected":
            weight = decay(now - event["created_at"], policy.rejection_half_life_days)
            weights[event["template_id"]] = min(3.0, weights.get(event["template_id"], 0.0) + weight)
    return weights


def rerank(ranking, feedback, catalogue, now, policy=Policy()):
    """ranking: model order of dicts with template_id, score (or None for popularity) and predicted_effort."""
    rejections = rejection_weights(feedback, now, policy)
    bias = effort_bias(feedback, now, policy)
    adjusted = []
    for position, item in enumerate(ranking):
        template = item["template_id"]
        score = item["score"]
        utility = math.log(max(score, 1e-12)) if score is not None else -math.log(position + 1)
        reasons = []
        if rejections.get(template) and policy.rejection_penalty:
            utility -= policy.rejection_penalty * rejections[template]
            reasons.append("lowered: you skipped this recently")
        shift = policy.effort_shift * bias * intensity(template, catalogue)
        if abs(shift) > 1e-9:
            utility += shift
            direction = "harder" if bias > 0 else "lighter"
            reasons.append(f"{'raised' if shift > 0 else 'lowered'}: recent sessions call for {direction} workouts")
        predicted = item.get("predicted_effort")
        if predicted is not None and policy.learned_effort_weight:
            utility -= policy.learned_effort_weight * 2 * abs(predicted - 0.5)
            if abs(predicted - 0.5) > 0.25:
                reasons.append("lowered: predicted " + ("too hard" if predicted > 0.5 else "too easy") + " for you")
        adjusted.append({**item, "model_position": position, "utility": utility, "reasons": reasons})
    # Stable sort keeps the model order among equal utilities.
    return sorted(adjusted, key=lambda item: -item["utility"])
