from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from build_game_documents import build_all_documents, BASE_DIR

from time import perf_counter

MAX_INDEX_GAMES = 50000


CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
BATCH_SIZE = 128


def sanitize_metadata(meta):
    clean = {}

    for key, value in meta.items():
        if value is None:
            clean[key] = -1

        elif isinstance(value, bool):
            clean[key] = bool(value)

        elif isinstance(value, int):
            clean[key] = int(value)

        elif isinstance(value, float):
            clean[key] = float(value)

        elif isinstance(value, str):
            clean[key] = value

        else:
            clean[key] = str(value)

    return clean

def main():
    total_start = perf_counter()

    print("Building game documents...")
    t0 = perf_counter()
    documents = build_all_documents()
    documents = documents[:MAX_INDEX_GAMES]  # keep this only if you are testing with a subset
    document_time = perf_counter() - t0

    print(f"Loaded {len(documents)} documents.")
    print(f"Document-building time: {document_time:.2f} seconds")

    print("Loading embedding model...")
    t0 = perf_counter()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model_time = perf_counter() - t0
    print(f"Embedding model load time: {model_time:.2f} seconds")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Embedding and indexing documents...")
    t0 = perf_counter()

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]

        ids = [item["appid"] for item in batch]
        texts = [item["document"] for item in batch]
        metadatas = [sanitize_metadata(item["metadata"]) for item in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"Indexed {min(start + BATCH_SIZE, len(documents))} / {len(documents)}")

    indexing_time = perf_counter() - t0
    total_time = perf_counter() - total_start

    print("\nDone.")
    print(f"Chroma index saved in: {CHROMA_PATH}")
    print("\nTiming summary:")
    print(f"Document-building time: {document_time:.2f} seconds")
    print(f"Model loading time:       {model_time:.2f} seconds")
    print(f"Embedding/indexing time:  {indexing_time:.2f} seconds")
    print(f"Total time:               {total_time:.2f} seconds")


if __name__ == "__main__":
    main()