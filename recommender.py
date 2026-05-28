from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
import ollama

from steam_sqlite import load_games_from_sqlite

# --- Paths and constants ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "steam_games"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma4:latest"

# Load enough games to cover the full Chroma index (~39k).
# If MAX_GAMES is lower than the indexed set, retrieved app_ids will be
# missing from self.records and results silently disappear.
MAX_GAMES = 50_000
DEFAULT_MATCH_COUNT = 5
CANDIDATE_POOL = 20  # retrieve N from vector DB, rank down to DEFAULT_MATCH_COUNT


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
                "retrieval_mode": "review-embeddings",
            },
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """Vector search over per-game review digests."""
        q_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=CANDIDATE_POOL)

        appids = res["ids"][0]
        distances = res["distances"][0]

        out: list[GameRecord] = []
        for appid, dist in zip(appids, distances):
            rec = self.by_appid.get(appid)
            if rec is None:
                continue  # Chroma knows this game but the games table didn't load it
            # Stash similarity on the record for the ranker to pick up.
            rec._retrieval_score = 1.0 - float(dist)  # cosine similarity
            out.append(rec)
        return out

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """MVP: trust Chroma's order. Re-ranker will go here later."""
        return [(rec, getattr(rec, "_retrieval_score", 0.0)) for rec in candidates]

    def _fetch_top_review(self, app_id: str) -> str:
        """Grab one short, high-quality review for grounding the LLM answer."""
        con = sqlite3.connect(self.db_path)
        try:
            row = con.execute("""
                SELECT review FROM reviews
                WHERE appid = ?
                  AND language = 'english'
                  AND LENGTH(review) BETWEEN 60 AND 300
                ORDER BY weighted_vote_score DESC, votes_up DESC
                LIMIT 1
            """, (app_id,)).fetchone()
        finally:
            con.close()
        return row[0] if row else ""

    def _build_context(self, matches: list[tuple[GameRecord, float]]) -> str:
        blocks = []
        for rec, _ in matches:
            tags = ", ".join(rec._normalize_tags(rec.raw.get("tags"))[:5])
            review = self._fetch_top_review(rec.app_id)
            blocks.append(
                f"GAME: {rec.name}\n"
                f"DESCRIPTION: {rec.short_description}\n"
                f"TAGS: {tags}\n"
                f"PLAYER REVIEW: {review}"
            )
        return "\n\n---\n\n".join(blocks)

    def generate_answer(
        self, query: str, matches: list[tuple[GameRecord, float]]
    ) -> str:
        if not matches:
            return "No matches found for your query."

        context = self._build_context(matches)
        prompt = (
            f"A user is looking for a game and asked: \"{query}\"\n\n"
            f"Here are the top candidate games retrieved from a database, "
            f"with one real player review each:\n\n"
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
            # Fall back to a simple answer if Ollama isn't running
            names = ", ".join(rec.name for rec, _ in matches[:3])
            return f"(LLM unavailable: {e}) Top picks: {names}."
