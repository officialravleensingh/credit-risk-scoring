import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_embedder = None
_index = None
_chunks = []


def _load_regulations(filepath: str) -> list[str]:
    with open(filepath, 'r') as f:
        text = f.read()
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]
    return sections


def _build_index(chunks: list[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index


def get_retriever():
    global _embedder, _index, _chunks

    if _index is not None:
        return _embedder, _index, _chunks

    regulations_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'regulations.txt')
    regulations_path = os.path.normpath(regulations_path)

    _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    _chunks = _load_regulations(regulations_path)
    _index = _build_index(_chunks, _embedder)

    return _embedder, _index, _chunks


def retrieve(query: str, k: int = 3) -> str:
    embedder, index, chunks = get_retriever()
    query_vec = embedder.encode([query], show_progress_bar=False).astype('float32')
    _, indices = index.search(query_vec, k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return '\n\n---\n\n'.join(results)
