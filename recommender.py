from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import json
import re

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import sqlite3
import ollama

from steam_sqlite import load_games_from_sqlite

# --- Paths and constants ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "steam_games"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "gemma4:latest"

MAX_GAMES = 50_000
DEFAULT_MATCH_COUNT = 5
CANDIDATE_POOL = 150  # retrieve N from vector DB, rank down to DEFAULT_MATCH_COUNT


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


# --- Data record -----------------------------------------------------------

@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


# --- Search engine ---------------------------------------------------------

class GameSearchEngine:
    """
    LLM-driven recommender backed by:
      - SQLite (games table) for display metadata
      - ChromaDB collection of per-game review-digest embeddings (MiniLM)
      - (Coming next) Ollama for the natural-language answer

    The JSON shape returned by `search()` is preserved so the Flask
    frontend keeps working.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # Load all games into memory for fast metadata lookup by app_id.
        self.records = self.load_records()
        self.by_appid = {r.app_id: r for r in self.records}

        # Embedding model + persistent vector DB built in the notebook.
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)

        # Embedding model + persistent vector DB built in the notebook.
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)

    def load_records(self) -> list[GameRecord]:
        return [
            GameRecord(app_id=app_id, raw=raw)
            for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES)
        ]

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked = self.rank_candidates(query, candidates)[:DEFAULT_MATCH_COUNT]
        results = [rec.to_result(score) for rec, score in ranked]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked),
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "game-document-embeddings",
                "filters": getattr(self, "_last_filters", {}),
            },
        }

    def _maybe_has_filter(self, query: str) -> bool:
        """Cheap gate: only call the LLM parser when the query looks constrained,
        so 'famous platformers' stays fast and pays no LLM round-trip."""
        q = query.lower()
        if any(ch.isdigit() for ch in q):
            return True
        keywords = (
            "under", "over", "below", "above", "cheap", "free", "budget",
            "mac", "macbook", "linux", "windows", "$", "€", "euro", "dollar", "USD"
            "before", "after", "newer", "older", "recent",
        )
        return any(k in q for k in keywords)

    def parse_query_filters(self, query: str) -> dict[str, Any]:
        """Use the LLM to extract structured constraints. Returns a dict that may
        contain max_price, min_price, platforms, min_year, max_year. {} on failure."""
        if not self._maybe_has_filter(query):
            return {}

        prompt = (
            "Extract search filters from this game search query. "
            "Respond with ONLY a JSON object, no prose, no markdown.\n"
            "Include a key ONLY if the query clearly implies it:\n"
            '  "max_price": number   (e.g. "under 20" -> 20)\n'
            '  "min_price": number   (e.g. "over 60" -> 60)\n'
            '  "platforms": array of any of "mac","linux","windows"\n'
            '  "min_year": integer   (e.g. "after 2020" -> 2020)\n'
            '  "max_year": integer   (e.g. "before 2015" -> 2015)\n'
            "If the query implies no constraints, return {}.\n\n"
            f'Query: "{query}"\nJSON:'
        )
        try:
            resp = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            text = resp["message"]["content"].strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Translate parsed filters into a Chroma `where` clause. Price/year use
        >= 0 / valid-value guards so 'unknown' rows (stored as -1) don't sneak in."""
        conditions: list[dict[str, Any]] = []

        max_price = filters.get("max_price")
        min_price = filters.get("min_price")
        if isinstance(max_price, (int, float)):
            conditions.append({"price": {"$gte": 0}})
            conditions.append({"price": {"$lte": float(max_price)}})
        if isinstance(min_price, (int, float)):
            conditions.append({"price": {"$gte": float(min_price)}})

        platforms = filters.get("platforms") or []
        if isinstance(platforms, str):
            platforms = [platforms]
        for p in platforms:
            if p in ("mac", "linux", "windows"):
                conditions.append({p: True})

        min_year = filters.get("min_year")
        max_year = filters.get("max_year")
        if isinstance(min_year, int):
            conditions.append({"release_year": {"$gte": min_year}})
        if isinstance(max_year, int):
            conditions.append({"release_year": {"$lte": max_year}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _chroma_query(self, q_emb, where):
        kwargs = dict(
            query_embeddings=q_emb,
            n_results=CANDIDATE_POOL,
            include=["documents", "metadatas", "distances"],
        )
        if where is not None:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        Retrieve candidate games from the new Chroma index.

        The Chroma index now contains one game-level document per game:
        title + description + genres + tags + categories + player feedback.
        """
        q_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()

        filters = self.parse_query_filters(query)
        where = self.build_where_clause(filters)
        self._last_filters = filters

        res = self._chroma_query(q_emb, where)

        # If the filter was too strict and nothing came back, retry unfiltered
        # so the user still gets recommendations instead of an empty list.
        if where is not None and not res["ids"][0]:
            res = self._chroma_query(q_emb, None)
            self._last_filters = {}

        appids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]

        out: list[GameRecord] = []

        for appid, document, metadata, dist in zip(appids, documents, metadatas, distances):
            rec = self.by_appid.get(appid)

            if rec is None:
                rec = self.by_appid.get(str(appid))

            if rec is None:
                try:
                    rec = self.by_appid.get(int(appid))
                except Exception:
                    rec = None

            if rec is None:
                continue

            rec._retrieval_score = 1.0 - float(dist)
            rec._retrieval_document = document
            rec._retrieval_metadata = metadata

            out.append(rec)

        return out

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Rerank candidates with a cross-encoder for sharp query-document
        relevance, then blend in popularity / quality / playtime priors.

        The cross-encoder reads (query, document) jointly, so it can tell
        'co-op farming sim' apart from 'co-op shooter' in a way cosine
        similarity over separate embeddings cannot.
        """

        if not candidates:
            return []

        def get_meta(rec: GameRecord) -> dict[str, Any]:
            return getattr(rec, "_retrieval_metadata", {}) or {}

        def safe_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        # 1. Cross-encoder relevance: score each (query, document) pair.
        pairs = [
            [query, getattr(rec, "_retrieval_document", "") or rec.name]
            for rec in candidates
        ]
        rerank_scores = self.reranker.predict(pairs)

        # ms-marco scores are unbounded logits; squash to (0, 1) so they
        # combine cleanly with the other [0, 1] signals.
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        relevance = [sigmoid(float(s)) for s in rerank_scores]

        # 2. Pool-normalised popularity / playtime priors (as before).
        max_log_recommendations = max(
            math.log1p(max(0.0, safe_float(get_meta(rec).get("recommendations"))))
            for rec in candidates
        ) or 1.0

        max_log_playtime = max(
            math.log1p(max(0.0, safe_float(get_meta(rec).get("average_playtime_forever"))))
            for rec in candidates
        ) or 1.0

        ranked = []

        for rec, rel in zip(candidates, relevance):
            meta = get_meta(rec)

            positive_ratio = max(0.0, safe_float(meta.get("positive_ratio")))
            recommendations = max(0.0, safe_float(meta.get("recommendations")))
            playtime = max(0.0, safe_float(meta.get("average_playtime_forever")))

            popularity_score = math.log1p(recommendations) / max_log_recommendations
            playtime_score = math.log1p(playtime) / max_log_playtime

            final_score = (
                0.80 * rel
                + 0.08 * popularity_score
                + 0.08 * positive_ratio
                + 0.04 * playtime_score
            )

            rec._relevance_score = rel
            rec._popularity_score = popularity_score
            rec._quality_score = positive_ratio
            rec._playtime_score = playtime_score

            ranked.append((rec, final_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def generate_answer(
        self, query: str, matches: list[tuple[GameRecord, float]]
    ) -> str:
        if not matches:
            return "No matches found for your query."

        context_blocks = []

        for rec, score in matches:
            document = getattr(rec, "_retrieval_document", "")

            context_blocks.append(
                f"GAME: {rec.name}\n"
                f"SIMILARITY SCORE: {score:.3f}\n"
                f"{document}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        prompt = (
            f"A user is looking for a game and asked: \"{query}\"\n\n"
            f"Here are the top candidate games retrieved from a database, "
            f"with game metadata and selected player feedback:\n\n"
            f"{context}\n\n"
            f"Write a friendly 2-3 sentence recommendation. Highlight which "
            f"of these games best matches the user's request and briefly say why, "
            f"grounding your reasoning in the descriptions and player reviews. "
            f"Do not invent games that are not in the list above."
        )

        try:
            resp = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4},
            )
            return resp["message"]["content"].strip()

        except Exception as e:
            names = ", ".join(rec.name for rec, _ in matches[:3])
            return f"(LLM unavailable: {e}) Top picks: {names}."