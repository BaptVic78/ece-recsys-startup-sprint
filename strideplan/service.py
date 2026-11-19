"""Recommendation service with feedback logging. Shared by the API and the simulator."""

import math
import random
import statistics
import threading
import time

from strideplan.reranker import Policy, rerank
from strideplan.store import FeedbackError


FEEDBACK_LOOKBACK_DAYS = 90


def heart_rate(record):
    try:
        value = float(record.get("avg_heart_rate_bpm"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 < value <= 300 else None


class StridePlanService:
    def __init__(self, recommender, store, base_histories=None, policy=Policy(), seed=None, clock=time.time):
        self.recommender = recommender
        self.store = store
        self.base_histories = base_histories or {}
        self.policy = policy
        self.clock = clock
        self.rng = random.Random(seed)
        self.rng_lock = threading.Lock()

    @property
    def model_version(self):
        return self.recommender.bundle.version

    def history(self, user_id):
        """Known Endomondo history followed by sessions logged through the API.

        Almost every training session has heart rate, so a missing value flags an unusual run to the GRU and
        shifts its predictions. Logged sessions without heart rate take the runner's recent median instead.
        """
        if user_id is None:
            return []
        known = list(self.base_histories.get(user_id, []))
        logged = self.store.sessions(user_id)
        recent = [value for value in (heart_rate(record) for record in (known + logged)[-30:]) if value is not None]
        if recent:
            typical = statistics.median(recent)
            logged = [{**record, "avg_heart_rate_bpm": typical} if heart_rate(record) is None else record
                      for record in logged]
        return known + logged

    def recent_feedback(self, user_id, now):
        if user_id is None:
            return []
        return self.store.feedback(user_id, since=now - FEEDBACK_LOOKBACK_DAYS * 86400, until=now + 1e-6)

    def recommend(self, user_id, max_duration_minutes, top_k, history=None, now=None, policy=None):
        """An inline history replaces stored sessions; the user's feedback still applies."""
        now = self.clock() if now is None else now
        policy = policy or self.policy
        records = list(history) if history is not None else self.history(user_id)
        ranked = self.recommender.rank(records)
        feedback = self.recent_feedback(user_id, now)
        adjusted = rerank(ranked["ranking"], feedback, self.recommender.catalogue, now, policy)
        eligible = self.recommender.within_budget(adjusted, max_duration_minutes)
        chosen = [{**item, "explored": False} for item in eligible[:top_k]]
        with self.rng_lock:
            explore = top_k >= 2 and len(eligible) > top_k and self.rng.random() < policy.epsilon
            pick = self.rng.choice(eligible[top_k:]) if explore else None
        if pick is not None:
            chosen[-1] = {**pick, "explored": True,
                          "reasons": pick["reasons"] + ["exploration: helps us learn what else suits you"]}
        for position, item in enumerate(chosen):
            item["position"] = position
        rec_id = self.store.log_recommendation(
            user_id, now, self.model_version, ranked["strategy"], max_duration_minutes,
            [{key: item[key] for key in ("position", "template_id", "score", "explored", "model_position")}
             for item in chosen])
        return {"rec_id": rec_id, "strategy": ranked["strategy"],
                "history_sessions_used": ranked["history_sessions_used"],
                "feedback_events_used": len(feedback),
                # Also true when feedback pushed a category out of the served list.
                "feedback_applied": any(item["reasons"] for item in adjusted),
                "recommendations": chosen}

    def record_feedback(self, rec_id, template_id, action, effort=None, session=None, now=None):
        """A completed workout may carry the session the user actually did; it then enters their history."""
        now = self.clock() if now is None else now
        if session is not None and action != "completed":
            raise FeedbackError("Only a completed workout can include a session")
        served = self.store.recommendation(rec_id)
        if session is not None and (served is None or served["user_id"] is None):
            raise FeedbackError("Sessions need a recommendation made for a user_id")
        feedback_id = self.store.add_feedback(rec_id, template_id, action, effort, now)
        logged = self.store.add_session(served["user_id"], session, now, rec_id) if session is not None else False
        return {"feedback_id": feedback_id, "session_logged": logged}

    def record_session(self, user_id, session, now=None):
        return self.store.add_session(user_id, session, self.clock() if now is None else now)
