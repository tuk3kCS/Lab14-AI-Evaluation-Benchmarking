"""Unit tests cho retrieval_eval và ragas_eval — Duyên."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.retrieval_eval import RetrievalEvaluator
from engine.ragas_eval import RAGASEvaluator


def test_hit_rate_and_mrr():
    ev = RetrievalEvaluator(default_top_k=3)

    assert ev.calculate_hit_rate(["doc_a"], ["x", "doc_a", "y"]) == 1.0
    assert ev.calculate_hit_rate(["doc_a"], ["x", "y", "z"]) == 0.0
    assert ev.calculate_mrr(["doc_a"], ["x", "doc_a"]) == 0.5
    assert ev.calculate_mrr(["doc_a"], ["doc_a"]) == 1.0
    assert ev.calculate_recall_at_k(["a", "b"], ["a", "x"], top_k=2) == 0.5


def test_out_of_context_expected_ids_empty():
    ev = RetrievalEvaluator()
    assert ev.calculate_hit_rate([], []) == 1.0
    assert ev.calculate_hit_rate([], ["doc_1"]) == 0.0


def test_evaluate_single_missed_ids():
    ev = RetrievalEvaluator(default_top_k=3)
    detail = ev.evaluate_single(["a", "b"], ["x", "y", "z"])
    assert detail["hit_rate"] == 0.0
    assert detail["missed_ids"] == ["a", "b"]
    assert detail["first_hit_rank"] is None


async def test_ragas_retrieval_miss_caps_faithfulness():
    ragas = RAGASEvaluator()
    score = await ragas.score(
        {
            "question": "Làm sao đổi mật khẩu?",
            "expected_answer": "Vào cài đặt tài khoản để đổi mật khẩu.",
            "expected_retrieval_ids": ["policy_pwd"],
        },
        {
            "answer": "Bạn có thể đổi mật khẩu bất cứ lúc nào qua email cá nhân.",
            "contexts": ["Thông tin không liên quan về thời tiết hôm nay."],
            "retrieved_ids": ["wrong_doc"],
        },
    )
    assert score["retrieval"]["hit_rate"] == 0.0
    assert score["faithfulness"] <= 0.35
    assert "Retrieval miss" in score["retrieval_answer_link"]


async def test_ragas_hit_improves_faithfulness():
    ragas = RAGASEvaluator()
    score = await ragas.score(
        {
            "question": "Hit Rate là gì?",
            "expected_answer": "Hit Rate đo tỉ lệ retrieve đúng trong top k.",
            "expected_retrieval_ids": ["eval_metrics"],
        },
        {
            "answer": "Hit Rate đo tỉ lệ retrieve đúng chunk trong top k.",
            "contexts": ["Hit Rate đo tỉ lệ retrieve đúng chunk trong top k của vector db."],
            "retrieved_ids": ["eval_metrics", "other"],
        },
    )
    assert score["retrieval"]["hit_rate"] == 1.0
    assert score["faithfulness"] > 0.5


if __name__ == "__main__":
    test_hit_rate_and_mrr()
    test_out_of_context_expected_ids_empty()
    test_evaluate_single_missed_ids()
    asyncio.run(test_ragas_retrieval_miss_caps_faithfulness())
    asyncio.run(test_ragas_hit_improves_faithfulness())
    print("All tests passed.")
