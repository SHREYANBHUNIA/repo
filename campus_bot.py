"""Campus Information Bot with compressed knowledge base retrieval."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_PATH = ROOT / "data" / "campus_index.json.gz"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "may",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
}


@dataclass
class Passage:
    source_id: str
    source_type: str
    title: str
    text: str


class CompressedCampusIndex:
    def __init__(self, index_path: Path = INDEX_PATH):
        self.index_path = index_path
        self.passages: List[Passage] = []
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            t.lower()
            for t in TOKEN_PATTERN.findall(text)
            if t.lower() not in STOPWORDS
        ]

    def build(self, course_file: Path, policy_file: Path) -> None:
        course_docs = json.loads(course_file.read_text())
        policy_docs = json.loads(policy_file.read_text())

        self.passages = [
            Passage(item["id"], "course_catalog", item["title"], item["text"])
            for item in course_docs
        ] + [
            Passage(item["id"], "campus_policy", item["title"], item["text"])
            for item in policy_docs
        ]

        documents = [self._tokenize(f"{p.title} {p.text}") for p in self.passages]
        self.idf = self._compute_idf(documents)
        self.doc_vectors = [self._tfidf_vector(tokens) for tokens in documents]
        self.doc_norms = [self._norm(vec) for vec in self.doc_vectors]

        payload = {
            "passages": [asdict(p) for p in self.passages],
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
        }
        compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        self.index_path.write_bytes(compressed)

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self.index_path}. Run with --build-index first."
            )
        payload = json.loads(gzip.decompress(self.index_path.read_bytes()).decode("utf-8"))
        self.passages = [Passage(**p) for p in payload["passages"]]
        self.idf = {k: float(v) for k, v in payload["idf"].items()}
        self.doc_vectors = [
            {k: float(v) for k, v in vec.items()} for vec in payload["doc_vectors"]
        ]
        self.doc_norms = [float(x) for x in payload["doc_norms"]]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Passage, float]]:
        query_tokens = self._tokenize(query)
        q_vec = self._tfidf_vector(query_tokens)
        q_norm = self._norm(q_vec)
        if q_norm == 0:
            return []

        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine(q_vec, q_norm, doc_vec, self.doc_norms[idx])
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.passages[idx], score) for idx, score in scores[:top_k]]

    def compression_stats(self) -> Dict[str, float]:
        raw_payload = {
            "passages": [asdict(p) for p in self.passages],
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
        }
        raw_bytes = len(json.dumps(raw_payload).encode("utf-8"))
        compressed_bytes = len(self.index_path.read_bytes()) if self.index_path.exists() else 0
        ratio = (compressed_bytes / raw_bytes) if raw_bytes else 1.0
        token_savings_estimate = int((1 - ratio) * 1000)
        return {
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": ratio,
            "token_savings_estimate": max(token_savings_estimate, 0),
        }

    @staticmethod
    def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
        n_docs = len(documents)
        df: Dict[str, int] = {}
        for doc in documents:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1
        return {tok: math.log((1 + n_docs) / (1 + freq)) + 1 for tok, freq in df.items()}

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        vec: Dict[str, float] = {}
        token_count = len(tokens)
        for tok, count in tf.items():
            if tok in self.idf:
                vec[tok] = (count / token_count) * self.idf[tok]
        return vec

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    @staticmethod
    def _cosine(
        vec_a: Dict[str, float],
        norm_a: float,
        vec_b: Dict[str, float],
        norm_b: float,
    ) -> float:
        if norm_a == 0 or norm_b == 0:
            return 0.0
        common = set(vec_a).intersection(vec_b)
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        return dot / (norm_a * norm_b)


class CampusInfoBot:
    def __init__(self, index: CompressedCampusIndex):
        self.index = index
        self.cache: Dict[str, Dict[str, object]] = {}

    def ask(self, question: str) -> Dict[str, object]:
        if question in self.cache:
            result = dict(self.cache[question])
            result["from_cache"] = True
            return result

        started = time.perf_counter()
        hits = self.index.search(question)
        latency_ms = (time.perf_counter() - started) * 1000

        if not hits:
            answer = (
                "I could not find a matching campus policy or course catalog entry. "
                "Please contact advising for authoritative guidance."
            )
            sources = []
        else:
            best = hits[0][0]
            answer = f"Best match: {best.text}"
            sources = [
                {
                    "id": doc.source_id,
                    "type": doc.source_type,
                    "title": doc.title,
                    "score": round(score, 4),
                    "snippet": doc.text,
                }
                for doc, score in hits
            ]

        stats = self.index.compression_stats()
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
            "estimated_token_savings": stats["token_savings_estimate"],
            "from_cache": False,
        }
        self.cache[question] = dict(result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campus information bot")
    parser.add_argument("--build-index", action="store_true", help="Build compressed search index")
    parser.add_argument("--ask", type=str, help="Question to ask the bot")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    index = CompressedCampusIndex()

    if args.build_index:
        index.build(DATA_DIR / "course_catalog.json", DATA_DIR / "campus_policies.json")
        stats = index.compression_stats()
        print(
            "Built compressed index:",
            f"raw={stats['raw_bytes']}B",
            f"compressed={stats['compressed_bytes']}B",
            f"ratio={stats['ratio']:.2f}",
        )

    if args.ask:
        if not index.index_path.exists():
            index.build(DATA_DIR / "course_catalog.json", DATA_DIR / "campus_policies.json")
        index.load()
        bot = CampusInfoBot(index)
        response = bot.ask(args.ask)
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
