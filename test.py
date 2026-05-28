from sentence_transformers import SentenceTransformer
from build_game_documents import build_all_documents

'''
We want to see what is a model that can handle our document's tokens.
Joaquin Roa - May 2026
'''

old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
new = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
print("old max tokens:", old.max_seq_length, "| new max tokens:", new.max_seq_length)

doc = build_all_documents()[0]["document"]
print("doc token length:", len(new.tokenizer(doc)["input_ids"]))