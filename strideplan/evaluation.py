"""Binary-relevance ranking metrics and a training-frequency baseline."""

import math
from collections import Counter
from statistics import mean


def ranking_metrics(ranking, relevant, k):
    """Recall and NDCG at k for one user, with unique item IDs.

    The denominator of Recall includes all relevant items. NDCG uses binary gains.
    Rankings shorter than k are valid. Empty relevance sets must be excluded by
    the evaluation protocol before calling this function.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Each evaluated user must have at least one relevant item")
    if len(ranking) != len(set(ranking)):
        raise ValueError("A ranking cannot contain duplicate items")
    gains = [item in relevant for item in ranking[:k]]
    recall = sum(gains) / len(relevant)
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    return {"recall": recall, "ndcg": dcg / ideal}


def evaluate(rankings, relevant_by_user, cutoffs=(1, 3)):
    """Macro-average over the same explicit set of users for all methods."""
    if not rankings or rankings.keys() != relevant_by_user.keys():
        raise ValueError("Rankings and targets must contain the same nonempty user set")
    if not cutoffs:
        raise ValueError("At least one cutoff is required")
    result = {"evaluated_users": len(rankings)}
    for k in cutoffs:
        values = [ranking_metrics(rankings[user], relevant_by_user[user], k) for user in rankings]
        result[f"recall@{k}"] = mean(value["recall"] for value in values)
        result[f"ndcg@{k}"] = mean(value["ndcg"] for value in values)
    return result


def most_popular(training_sequences, candidate_ids):
    """Pass training template IDs only. Break equal-frequency ties by item ID."""
    counts = Counter(item for sequence in training_sequences for item in sequence)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs must be unique")
    return sorted(candidate_ids, key=lambda item: (-counts[item], item))
