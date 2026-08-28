"""
retriever.py — Basic single-agent retriever (Week 1 baseline).

Queries both the code and docs Chroma collections and merges results by
similarity score into one ranked list. This is deliberately the "naive"
retrieval baseline: no routing by query type, no per-source retriever
logic — just similarity search across everything, re-ranked together.

This exists on purpose, not as a placeholder to be deleted: it's the
"before" side of the Week 4 ablation ("single merged retriever vs. router
+ multi-retriever architecture"). Week 2 will add a router-based
multi-retriever path in graph/ that calls the same underlying
collections differently; this file should stay as-is as the comparison
point rather than being rewritten in place.
"""

from dataclasses import dataclass

from repomind.ingestion.vectorstore import CODE_COLLECTION, DOCS_COLLECTION, get_vectorstore


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its similarity score, source-agnostic."""

    content: str
    metadata: dict
    score: float  # Chroma's relevance score; lower = more similar (L2 distance by default)
    source_collection: str  # CODE_COLLECTION or DOCS_COLLECTION


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query, searching both
    the code and docs collections and merging by score.

    Args:
        query: The user's natural-language question.
        k: Total number of chunks to return after merging (not per-collection).

    Returns:
        Up to k RetrievedChunk objects, sorted by relevance (best first).
        May return fewer than k if the collections together hold fewer
        than k documents.
    """
    code_store = get_vectorstore(CODE_COLLECTION)
    docs_store = get_vectorstore(DOCS_COLLECTION)

    # Over-fetch k from each collection before merging, since the top-k
    # overall could in principle come entirely from one collection.
    code_results = code_store.similarity_search_with_score(query, k=k)
    docs_results = docs_store.similarity_search_with_score(query, k=k)

    merged: list[RetrievedChunk] = []
    for doc, score in code_results:
        merged.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                source_collection=CODE_COLLECTION,
            )
        )
    for doc, score in docs_results:
        merged.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                source_collection=DOCS_COLLECTION,
            )
        )

    # Chroma's default distance metric is L2 (lower = more similar), so
    # ascending sort puts the best matches first. If the collections are
    # ever reconfigured for cosine/IP similarity (higher = better), this
    # sort direction would need to flip — flagging here rather than
    # silently getting it wrong later.
    merged.sort(key=lambda c: c.score)

    return merged[:k]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM
    prompt, labeling each chunk with its source so the model (and the
    later critique node) can attribute claims back to a specific file/
    section rather than treating the context as one undifferentiated blob.

    Args:
        chunks: Chunks returned by retrieve().

    Returns:
        A formatted string, one labeled block per chunk.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.source_collection == CODE_COLLECTION:
            label = f"[Code: {chunk.metadata.get('source')} — {chunk.metadata.get('name')}]"
        else:
            label = f"[Docs: {chunk.metadata.get('source')} — {chunk.metadata.get('section') or chunk.metadata.get('title')}]"
        blocks.append(f"{label}\n{chunk.content}")

    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    results = retrieve("How are passwords hashed?", k=5)
    for r in results:
        print(f"[{r.score:.4f}] ({r.source_collection}) {r.metadata.get('name') or r.metadata.get('section')}")

    print("\n--- Formatted context ---\n")
    print(format_context(results))