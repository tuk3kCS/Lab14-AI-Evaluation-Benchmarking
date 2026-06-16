import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "documents.json"

ADVERSARIAL_KEYWORDS = [
    "ignore", "bỏ qua", "hacker", "override", "jailbreak", "dan",
    "system override", "admin]", "xuất toàn bộ",
]
OUT_OF_CONTEXT_SIGNALS = [
    "cổ phiếu", "thời tiết", "nuôi mèo", "cohen", "kappa",
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


class MainAgent:
    """
    RAG Agent với 2 phiên bản:
    - v1: retrieval yếu, câu trả lời generic (baseline)
    - v2: keyword retrieval + guardrails + trả lời từ context
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.name = f"SupportAgent-{version}"
        self.corpus = self._load_corpus()
        self._latency_factor = 0.08 if version == "v1" else 0.04

    def _load_corpus(self) -> List[Dict]:
        if CORPUS_PATH.exists():
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["documents"]
        return []

    def _score_doc(self, question: str, doc: Dict) -> float:
        q_tokens = set(_tokenize(question))
        d_tokens = set(_tokenize(doc["content"] + " " + doc.get("section", "")))
        if not q_tokens:
            return 0.0
        return len(q_tokens & d_tokens) / len(q_tokens)

    def _retrieve_v1(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.corpus:
            return []
        return self.corpus[:top_k]

    def _retrieve_v2(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.corpus:
            return []
        scored = sorted(
            self.corpus,
            key=lambda d: self._score_doc(question, d),
            reverse=True,
        )
        return [d for d in scored if self._score_doc(question, d) > 0][:top_k] or scored[:1]

    def _is_adversarial(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in ADVERSARIAL_KEYWORDS)

    def _is_out_of_context(self, question: str) -> bool:
        q = question.lower()
        return any(sig in q for sig in OUT_OF_CONTEXT_SIGNALS)

    def _generate_v1(self, question: str, docs: List[Dict]) -> str:
        return (
            f"Dựa trên tài liệu hệ thống, tôi xin trả lời câu hỏi '{question[:80]}' "
            "như sau: [Câu trả lời mẫu]."
        )

    def _generate_v2(self, question: str, docs: List[Dict]) -> str:
        if self._is_out_of_context(question):
            return (
                "Tôi không tìm thấy thông tin liên quan trong tài liệu hệ thống. "
                "Tôi chỉ trả lời dựa trên tài liệu nội bộ được cung cấp."
            )

        if self._is_adversarial(question):
            if docs:
                content = docs[0]["content"]
                return (
                    "Tôi không thể thực hiện yêu cầu vi phạm chính sách bảo mật. "
                    f"Theo tài liệu: {content[:200]}"
                )
            return "Tôi không thể bỏ qua hướng dẫn bảo mật. Vui lòng liên hệ IT Support."

        if not docs:
            return "Tôi không có đủ thông tin trong tài liệu để trả lời câu hỏi này."

        primary = docs[0]["content"]
        if len(docs) > 1:
            secondary = docs[1]["content"][:150]
            return f"{primary} {secondary}"
        return primary

    def _estimate_tokens(self, text: str) -> int:
        return max(50, len(text.split()) * 2)

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(self._latency_factor)

        retrieve_fn = self._retrieve_v1 if self.version == "v1" else self._retrieve_v2
        docs = retrieve_fn(question, top_k=3)
        contexts = [d["content"] for d in docs]
        retrieved_ids = [d["id"] for d in docs]

        generate_fn = self._generate_v1 if self.version == "v1" else self._generate_v2
        answer = generate_fn(question, docs)

        tokens = self._estimate_tokens(question + answer + " ".join(contexts))
        cost_per_1k = 0.00015 if self.version == "v2" else 0.0002

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "gpt-4o-mini" if self.version == "v2" else "gpt-3.5-turbo",
                "tokens_used": tokens,
                "cost_usd": round(tokens / 1000 * cost_per_1k, 6),
                "sources": list({d.get("source", "unknown") for d in docs}),
                "version": self.version,
            },
        }


def create_agent(version: str = "v1") -> MainAgent:
    return MainAgent(version=version)
