import re
from typing import Dict, List

from engine.retrieval_eval import RetrievalEvaluator


def _tokenize(text: str) -> set:
    words = re.findall(r"[\wÀ-ỹ]+", text.lower())
    return {w for w in words if len(w) > 2}


def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class RAGASEvaluator:
    """Đánh giá faithfulness, relevancy và retrieval metrics."""

    def __init__(self, top_k: int = 3):
        self.retrieval = RetrievalEvaluator()
        self.top_k = top_k

    def _faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.3 if "không" in answer.lower() or "khong" in answer.lower() else 0.1
        combined = " ".join(contexts)
        return min(1.0, _overlap_ratio(answer, combined) * 2.5)

    def _relevancy(self, answer: str, expected: str, question: str) -> float:
        exp_score = _overlap_ratio(answer, expected)
        q_score = _overlap_ratio(answer, question) * 0.3
        return min(1.0, exp_score * 1.8 + q_score)

    async def score(self, case: Dict, response: Dict) -> Dict:
        answer = response.get("answer", "")
        contexts = response.get("contexts", [])
        retrieved_ids = response.get("retrieved_ids", [])
        expected_ids = case.get("expected_retrieval_ids", [])

        hit_rate = self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, self.top_k)
        mrr = self.retrieval.calculate_mrr(expected_ids, retrieved_ids)

        return {
            "faithfulness": round(self._faithfulness(answer, contexts), 3),
            "relevancy": round(
                self._relevancy(answer, case.get("expected_answer", ""), case.get("question", "")),
                3,
            ),
            "retrieval": {"hit_rate": hit_rate, "mrr": round(mrr, 3)},
        }

    async def evaluate_batch(self, cases: List[Dict], responses: List[Dict]) -> Dict:
        scores = []
        for case, resp in zip(cases, responses):
            scores.append(await self.score(case, resp))

        n = len(scores) or 1
        return {
            "avg_faithfulness": sum(s["faithfulness"] for s in scores) / n,
            "avg_relevancy": sum(s["relevancy"] for s in scores) / n,
            "avg_hit_rate": sum(s["retrieval"]["hit_rate"] for s in scores) / n,
            "avg_mrr": sum(s["retrieval"]["mrr"] for s in scores) / n,
        }
