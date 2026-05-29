# Steam Game Recommender (RAG)

An LLM-driven recommender system for Steam games, built for **Advanced Analytics in Business [D0S07a], Assignment 3** by Tarik Kalai and Joaquin Roa. Given a natural-language request (e.g. *"a chill cozy farming game with co-op"*), it retrieves relevant games, reranks them, and returns a short grounded recommendation through a Flask web app.

## How it works

The system is a retrieval-augmented generation (RAG) pipeline over the provided Steam dataset:

1. **Document building** One document per game (genres, tags, categories, description, and a digest of player reviews). Factual fields (price, platforms, release year, review counts) are kept as metadata.
2. **Embedding + indexing** — documents are embedded with `multi-qa-MiniLM-L6-cos-v1` and stored in a **ChromaDB** collection.
3. **Retrieval** — the query is embedded and used to pull a broad candidate pool from Chroma. If the query implies constraints (price / platform / year), a lightweight LLM step parses them into a Chroma `where` filter applied before search.
4. **Reranking** — candidates are rescored with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), then blended with popularity / quality / playtime priors.
5. **Answer generation** — the top games are passed to a local **Ollama** model (Gemma) that writes a short recommendation grounded in the retrieved descriptions and reviews.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and a local [Ollama](https://ollama.com/download) install.

```bash
# 1. Install dependencies
uv sync

# 2. Pull the LLM (run once)
ollama pull gemma4      # LLM_MODEL_NAME in recommender.py

# 3. Build the vector index (one-time, a few minutes on CPU)
uv run python build_chroma_index.py

# 4. Run the app
uv run python app.py
```

Then open <http://127.0.0.1:5000>.

> **Note:** the `.sqlite` dataset is **not** included in this repo and the `chroma_db/` index is regenerated locally by step 3.

## File overview

| File | Purpose |
| --- | --- |
| `recommender.py` | Core engine: retrieval, filtering, reranking, answer generation |
| `build_game_documents.py` | Builds one game-level document + metadata per game |
| `build_chroma_index.py` | Embeds the documents and writes the ChromaDB index |
| `test_chroma_query.py` | Standalone script to sanity-check raw vector retrieval |
| `app.py` | Necessary for running the project, unmodified from the baseline |
| `pyproject.toml` | Necessary for running the project, unmodified from the baseline |
| `steam_sqlite.py` | Necessary for running the project, unmodified from the baseline |
| `uv.lock` | Necessary for running the project, unmodified from the baseline |

Some other functionalities such as the `sqlite` database may be needed to be downloaded from the original source.