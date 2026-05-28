from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "steam_games"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

def search(query, n_results=10):
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery: {query}\n")

    for i, (appid, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        similarity = 1 - distance

        print(f"{i}. {metadata.get('name')} | appid={appid} | score={similarity:.3f}")
        print(f"   price={metadata.get('price')} | mac={metadata.get('mac')} | linux={metadata.get('linux')}")
        print()


if __name__ == "__main__":
    search("chill cozy farming game with co-op")