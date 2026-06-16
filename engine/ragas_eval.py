"""
RAGAS-style evaluation: faithfulness, relevancy và retrieval metrics tích hợp.

Module phụ trách: Duyên — Retrieval Engineer

Thiết kế: heuristic nhanh, không bắt buộc API — phù hợp chạy 50+ cases async.
Mối liên hệ Retrieval ↔ Answer: faithfulness phạt nặng khi hit_rate = 0 vì
agent không có context đúng để bám vào.
"""

from __future__ import annotations

import re
from typing import Dict, List

from engine.retrieval_eval import RetrievalEvaluator

_REFUSAL_PATTERNS = (
    r"không (?:có|có thông tin|tìm thấy|nằm trong)",
    r"khong (?:co|tim thay)",
    r"ngoài phạm vi",
    r"out of scope",
    r"không thể trả lời",
)


def _tokenize(text: str) -> set:
    words = re.findall(r"[\wÀ-ỹ]+", text.lower())
    return {w for w in words if len(w) > 2}


def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(re.search(p, lower) for p in _REFUSAL_PATTERNS)


class RAGASEvaluator:
    """Đánh giá faithfulness, relevancy và retrieval metrics cho từng test case."""

    def __init__(self, top_k: int = 3):
        self.retrieval = RetrievalEvaluator(default_top_k=top_k)
        self.top_k = top_k

    def _faithfulness(self, answer: str, contexts: List[str], hit_rate: float) -> float:
        """
        Faithfulness: mức câu trả lời bám vào context đã retrieve.

        - hit_rate = 0 → cap tối đa 0.35 (retrieval miss thường dẫn hallucination)
        - refusal hợp lệ khi không có context → cho điểm cao hơn generic answer
        """
        if not contexts:
            if _is_refusal(answer):
                return 0.85
            return 0.3 if "không" in answer.lower() or "khong" in answer.lower() else 0.1

        combined = " ".join(contexts)
        overlap = _overlap_ratio(answer, combined)
        base = min(1.0, overlap * 2.5)

        if hit_rate == 0.0:
            return round(min(base, 0.35), 3)

        return round(base, 3)

    def _relevancy(self, answer: str, expected: str, question: str) -> float:
        """Relevancy: overlap với expected_answer (trọng số cao) + question (trọng số thấp)."""
        exp_score = _overlap_ratio(answer, expected)
        q_score = _overlap_ratio(answer, question) * 0.3
        return round(min(1.0, exp_score * 1.8 + q_score), 3)

    def _context_precision(self, question: str, contexts: List[str]) -> float:
        """Heuristic context precision: context có overlap với câu hỏi không."""
        if not contexts:
            return 0.0
        scores = [_overlap_ratio(question, ctx) for ctx in contexts]
        return round(sum(scores) / len(scores), 3)

    def explain_retrieval_answer_link(self, retrieval_detail: Dict, faithfulness: float) -> str:
        """Giải thích ngắn mối liên hệ retrieval miss ↔ faithfulness thấp."""
        if not retrieval_detail.get("has_ground_truth"):
            return "Case out-of-context: không yêu cầu retrieval ground truth."

        if retrieval_detail["hit_rate"] == 0:
            return (
                f"Retrieval miss (thiếu {retrieval_detail['missed_ids']}) → "
                f"faithfulness bị cap thấp ({faithfulness})."
            )

        rank = retrieval_detail.get("first_hit_rank")
        return (
            f"Retrieval hit tại rank {rank} (MRR={retrieval_detail['mrr']}) → "
            f"faithfulness={faithfulness}."
        )

    async def score(self, case: Dict, response: Dict) -> Dict:
        answer = response.get("answer", "")
        contexts = response.get("contexts", [])
        retrieved_ids = response.get("retrieved_ids", [])
        expected_ids = case.get("expected_retrieval_ids", [])

        retrieval_detail = self.retrieval.evaluate_single(
            expected_ids, retrieved_ids, self.top_k
        )
        hit_rate = retrieval_detail["hit_rate"]
        mrr = retrieval_detail["mrr"]

        faithfulness = self._faithfulness(answer, contexts, hit_rate)
        relevancy = self._relevancy(
            answer, case.get("expected_answer", ""), case.get("question", "")
        )

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "context_precision": self._context_precision(case.get("question", ""), contexts),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": round(mrr, 3),
                "recall_at_k": retrieval_detail["recall_at_k"],
                "first_hit_rank": retrieval_detail["first_hit_rank"],
                "missed_ids": retrieval_detail["missed_ids"],
            },
            "retrieval_answer_link": self.explain_retrieval_answer_link(
                retrieval_detail, faithfulness
            ),
        }

    async def evaluate_batch(self, cases: List[Dict], responses: List[Dict]) -> Dict:
        scores = []
        for case, resp in zip(cases, responses):
            scores.append(await self.score(case, resp))

        n = len(scores) or 1
        retrieval_batch = await self.retrieval.evaluate_batch(cases, responses, self.top_k)

        return {
            "avg_faithfulness": sum(s["faithfulness"] for s in scores) / n,
            "avg_relevancy": sum(s["relevancy"] for s in scores) / n,
            "avg_context_precision": sum(s["context_precision"] for s in scores) / n,
            "avg_hit_rate": sum(s["retrieval"]["hit_rate"] for s in scores) / n,
            "avg_mrr": sum(s["retrieval"]["mrr"] for s in scores) / n,
            "retrieval_summary": retrieval_batch,
        }
