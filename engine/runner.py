import asyncio
import time
from typing import Dict, List

from engine.ragas_eval import RAGASEvaluator
from engine.llm_judge import MultiModelJudge

PASS_SCORE_THRESHOLD = 3.0


class BenchmarkRunner:
    def __init__(self, agent, evaluator=None, judge=None, batch_size: int = 10):
        self.agent = agent
        self.evaluator = evaluator or RAGASEvaluator()
        self.judge = judge or MultiModelJudge()
        self.batch_size = batch_size
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        ragas_scores, judge_result = await asyncio.gather(
            self.evaluator.score(test_case, response),
            self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case["expected_answer"],
            ),
        )

        meta = response.get("metadata", {})
        agent_tokens = meta.get("tokens_used", 0)
        agent_cost = meta.get("cost_usd", 0.0)
        judge_cost = judge_result.get("judge_cost_usd", 0.0)
        total_cost = agent_cost + judge_cost

        self.total_tokens += agent_tokens + judge_result.get("judge_tokens", 0)
        self.total_cost_usd += total_cost

        return {
            "test_case": test_case["question"],
            "test_meta": test_case.get("metadata", {}),
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "retrieved_ids": response.get("retrieved_ids", []),
            "agent_response": response["answer"],
            "latency": round(latency, 4),
            "tokens": agent_tokens + judge_result.get("judge_tokens", 0),
            "cost_usd": round(total_cost, 6),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "pass" if judge_result["final_score"] >= PASS_SCORE_THRESHOLD else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = None) -> List[Dict]:
        batch_size = batch_size or self.batch_size
        results = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            batch_results = await asyncio.gather(*[self.run_single_test(case) for case in batch])
            results.extend(batch_results)
            print(f"  ... {min(i + batch_size, len(dataset))}/{len(dataset)} cases")

        return results
