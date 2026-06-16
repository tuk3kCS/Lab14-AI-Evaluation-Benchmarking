"""
Retrieval evaluation metrics: Hit Rate @K và Mean Reciprocal Rank (MRR).

Module phụ trách: Duyên — Retrieval Engineer
"""

from __future__ import annotations

from typing import Dict, List, Optional


class RetrievalEvaluator:
    """Đánh giá chất lượng retrieval bằng cách đối chiếu retrieved_ids với ground truth."""

    def __init__(self, default_top_k: int = 3):
        self.default_top_k = default_top_k

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: Optional[int] = None,
    ) -> float:
        """
        Hit Rate @K: 1.0 nếu ít nhất một expected_id nằm trong top_k retrieved, ngược lại 0.0.

        Case không có expected_ids (ví dụ out-of-context cố ý) được coi là pass retrieval
        khi agent không retrieve gì; nếu vẫn retrieve thì 0.0.
        """
        top_k = top_k if top_k is not None else self.default_top_k
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0

        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR = 1 / rank (1-indexed) của expected_id đầu tiên xuất hiện trong retrieved_ids.
        Trả về 0.0 nếu không tìm thấy hoặc không có expected_ids.
        """
        if not expected_ids:
            return 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_recall_at_k(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: Optional[int] = None,
    ) -> float:
        """Recall@K = |expected ∩ top_k| / |expected|."""
        top_k = top_k if top_k is not None else self.default_top_k
        if not expected_ids:
            return 1.0

        top_retrieved = set(retrieved_ids[:top_k])
        hits = sum(1 for doc_id in expected_ids if doc_id in top_retrieved)
        return hits / len(expected_ids)

    def evaluate_single(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: Optional[int] = None,
    ) -> Dict:
        """Metrics chi tiết cho một case."""
        top_k = top_k if top_k is not None else self.default_top_k
        top_retrieved = retrieved_ids[:top_k]

        first_hit_rank = None
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                first_hit_rank = i + 1
                break

        missed_ids = [doc_id for doc_id in expected_ids if doc_id not in top_retrieved]

        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k),
            "mrr": round(self.calculate_mrr(expected_ids, retrieved_ids), 4),
            "recall_at_k": round(
                self.calculate_recall_at_k(expected_ids, retrieved_ids, top_k), 4
            ),
            "first_hit_rank": first_hit_rank,
            "missed_ids": missed_ids,
            "top_k": top_k,
            "has_ground_truth": bool(expected_ids),
        }

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        responses: List[Dict],
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Tổng hợp retrieval metrics trên toàn bộ benchmark run.
        Trả về avg metrics + danh sách case miss để phục vụ failure analysis.
        """
        top_k = top_k if top_k is not None else self.default_top_k
        per_case: List[Dict] = []
        hit_rates: List[float] = []
        mrrs: List[float] = []
        recalls: List[float] = []

        for case, resp in zip(dataset, responses):
            expected = case.get("expected_retrieval_ids", [])
            retrieved = resp.get("retrieved_ids", [])
            detail = self.evaluate_single(expected, retrieved, top_k)

            per_case.append(
                {
                    "question": case.get("question", "")[:120],
                    "type": case.get("metadata", {}).get("type", "unknown"),
                    **detail,
                }
            )
            hit_rates.append(detail["hit_rate"])
            mrrs.append(detail["mrr"])
            recalls.append(detail["recall_at_k"])

        n = len(hit_rates) or 1
        with_gt = [c for c in per_case if c["has_ground_truth"]]
        misses = [c for c in with_gt if c["hit_rate"] == 0]

        return {
            "avg_hit_rate": sum(hit_rates) / n,
            "avg_mrr": sum(mrrs) / n,
            "avg_recall_at_k": sum(recalls) / n,
            "total_cases": n,
            "cases_with_ground_truth": len(with_gt),
            "retrieval_miss_count": len(misses),
            "retrieval_miss_rate": len(misses) / (len(with_gt) or 1),
            "missed_cases": misses[:10],
            "per_case": per_case,
        }

    def compare_versions(
        self, v1_batch: Dict, v2_batch: Dict
    ) -> Dict:
        """So sánh delta retrieval giữa hai phiên bản agent (phục vụ regression)."""
        return {
            "hit_rate_delta": v2_batch["avg_hit_rate"] - v1_batch["avg_hit_rate"],
            "mrr_delta": v2_batch["avg_mrr"] - v1_batch["avg_mrr"],
            "recall_at_k_delta": v2_batch["avg_recall_at_k"] - v1_batch["avg_recall_at_k"],
            "miss_count_delta": v2_batch["retrieval_miss_count"]
            - v1_batch["retrieval_miss_count"],
        }
