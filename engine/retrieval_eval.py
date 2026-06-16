from typing import Dict, List


class RetrievalEvaluator:
    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3
    ) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict], responses: List[Dict]) -> Dict:
        hit_rates, mrrs = [], []
        for case, resp in zip(dataset, responses):
            expected = case.get("expected_retrieval_ids", [])
            retrieved = resp.get("retrieved_ids", [])
            hit_rates.append(self.calculate_hit_rate(expected, retrieved))
            mrrs.append(self.calculate_mrr(expected, retrieved))

        n = len(hit_rates) or 1
        return {
            "avg_hit_rate": sum(hit_rates) / n,
            "avg_mrr": sum(mrrs) / n,
            "total_cases": n,
        }
