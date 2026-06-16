from typing import Dict, Tuple


DEFAULT_THRESHOLDS = {
    "min_score_delta": 0.05,
    "min_hit_rate_delta": 0.0,
    "max_latency_regression_pct": 0.25,
    "max_cost_regression_pct": 0.30,
    "min_avg_score": 2.5,
}


def evaluate_release_gate(
    v1: Dict,
    v2: Dict,
    thresholds: Dict = None,
) -> Tuple[str, Dict]:
    """
    Quyết định APPROVE hoặc BLOCK RELEASE dựa trên delta chất lượng/chi phí/hiệu năng.
  """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    m1, m2 = v1["metrics"], v2["metrics"]

    delta = {
        "avg_score": m2.get("avg_score", 0) - m1.get("avg_score", 0),
        "hit_rate": m2.get("hit_rate", 0) - m1.get("hit_rate", 0),
        "agreement_rate": m2.get("agreement_rate", 0) - m1.get("agreement_rate", 0),
        "avg_latency": m2.get("avg_latency", 0) - m1.get("avg_latency", 0),
        "avg_cost_usd": m2.get("avg_cost_usd", 0) - m1.get("avg_cost_usd", 0),
        "avg_mrr": m2.get("avg_mrr", 0) - m1.get("avg_mrr", 0),
    }

    latency_pct = (
        delta["avg_latency"] / m1["avg_latency"]
        if m1.get("avg_latency", 0) > 0
        else 0
    )
    cost_pct = (
        delta["avg_cost_usd"] / m1["avg_cost_usd"]
        if m1.get("avg_cost_usd", 0) > 0
        else 0
    )

    reasons = []
    blockers = []

    if m2.get("avg_score", 0) < t["min_avg_score"]:
        blockers.append(f"avg_score V2 ({m2['avg_score']:.2f}) < ngưỡng {t['min_avg_score']}")

    if delta["avg_score"] < t["min_score_delta"]:
        blockers.append(f"score delta ({delta['avg_score']:+.2f}) < {t['min_score_delta']}")

    if delta["hit_rate"] < t["min_hit_rate_delta"]:
        blockers.append(f"hit_rate delta ({delta['hit_rate']:+.2f}) âm")

    if latency_pct > t["max_latency_regression_pct"]:
        blockers.append(f"latency tăng {latency_pct*100:.1f}% > {t['max_latency_regression_pct']*100:.0f}%")

    if cost_pct > t["max_cost_regression_pct"]:
        blockers.append(f"cost tăng {cost_pct*100:.1f}% > {t['max_cost_regression_pct']*100:.0f}%")

    if not blockers:
        reasons.append("Tất cả ngưỡng regression đạt yêu cầu")
        decision = "APPROVE"
    elif delta["avg_score"] >= 0.3 and delta["hit_rate"] >= 0:
        reasons.append("Cải thiện chất lượng đáng kể (+0.3 score) — override latency/cost nhẹ")
        decision = "APPROVE"
    else:
        decision = "BLOCK"

    return decision, {
        "decision": decision,
        "delta": delta,
        "latency_regression_pct": round(latency_pct, 3),
        "cost_regression_pct": round(cost_pct, 3),
        "blockers": blockers,
        "reasons": reasons,
    }
