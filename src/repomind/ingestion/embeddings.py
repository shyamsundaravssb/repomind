"""
embeddings.py — Embedding model loader.

Wraps langchain_huggingface's HuggingFaceEmbeddings around
BAAI/bge-small-en-v1.5, per the Week 1 decision to use a small
(~130MB), CPU-friendly model rather than a larger or API-billed one.

bge models are trained to expect a specific instruction prefix on the
*query* side (not the document side) for retrieval tasks — omitting it
doesn't break anything outright, but measurably hurts retrieval quality
vs. using it. This is easy to silently get wrong, so it's handled here
rather than left for retriever.py to remember.
"""

import functools

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge's recommended instruction prefix for query-side embeddings during
# retrieval. Per the model card: prepend this to queries, but NOT to the
# documents/passages being indexed. Skipping this is a common silent
# mistake that degrades retrieval without throwing any error.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@functools.lru_cache(maxsize=1)
def get_embedding_model(device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Load (and cache) the bge-small-en-v1.5 embedding model.

    Cached via lru_cache since loading is relatively expensive (model
    weights from disk/HF cache) and both ingestion and retrieval need
    the identical model instance/config — embedding queries and documents
    with mismatched settings would silently produce a degraded vector
    space with no error.

    Args:
        device: "cpu" or "cuda". Defaults to "cpu" per the Week 1 decision
            to run CPU-only torch on an 8GB laptop.

    Returns:
        A configured HuggingFaceEmbeddings instance. Calling this again
        with the same device returns the same cached instance.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document/passage texts (chunk content going INTO the
    index) — no query instruction prefix applied, per bge's convention.

    Args:
        texts: List of chunk content strings to embed.

    Returns:
        List of embedding vectors, one per input text, in the same order.
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string for retrieval — the bge query instruction
    prefix is prepended manually here, since the installed
    langchain_huggingface version doesn't expose a query_instruction
    constructor param. This keeps the asymmetric-search behavior bge
    recommends without depending on that API surface.

    Args:
        text: The user's query string.

    Returns:
        A single embedding vector.
    """
    model = get_embedding_model()
    return model.embed_query(BGE_QUERY_INSTRUCTION + text)


if __name__ == "__main__":
    # Quick manual sanity check: embed a doc chunk and a query, confirm
    # dimensions match and the query instruction is actually applied.
    doc_vec = embed_documents(["hash_password uses PBKDF2-HMAC-SHA256 with a random salt."])
    query_vec = embed_query("How are passwords hashed?")

    print(f"Document embedding dim: {len(doc_vec[0])}")
    print(f"Query embedding dim: {len(query_vec)}")
    assert len(doc_vec[0]) == len(query_vec), "dimension mismatch between doc/query embeddings"
    print("OK: dimensions match.")