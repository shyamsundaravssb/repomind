"""
vectorstore.py — Chroma vector store init/persist logic.

Wraps langchain_chroma's Chroma class with two concerns specific to this
project: (1) converting CodeChunk/DocChunk dataclasses into LangChain
Documents with the right metadata, and (2) using separate Chroma
*collections* per source type (code vs docs) rather than one merged
collection — this mirrors the spec's "separate retrievers per source
type" architecture (Week 2), so the vector store layer needs to support
that split from the start rather than being retrofitted later.

Persistence: Chroma is configured with persist_directory=data/chroma_db
(gitignored per the folder structure), so the index survives across
process restarts without re-embedding on every run.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from repomind.ingestion.code_chunker import CodeChunk
from repomind.ingestion.doc_chunker import DocChunk
from repomind.ingestion.embeddings import get_embedding_model

DEFAULT_PERSIST_DIR = Path("data/chroma_db")

# Separate collections per source type — supports the router/multi-retriever
# architecture in Week 2 without needing a schema change later.
CODE_COLLECTION = "code_chunks"
DOCS_COLLECTION = "doc_chunks"


def _chunks_to_documents(chunks: list[CodeChunk] | list[DocChunk]) -> list[Document]:
    """
    Convert a list of CodeChunk or DocChunk objects into LangChain
    Documents, using each chunk's own to_document_metadata() so this
    function stays agnostic to which chunk type it's given.

    Args:
        chunks: A list of CodeChunk or DocChunk objects (must not be mixed
            — Chroma metadata schemas are per-collection, and code/doc
            chunks have different metadata shapes).

    Returns:
        A list of Document objects ready to add to a Chroma collection.

    Raises:
        ValueError: If chunks is empty (nothing meaningful to convert).
    """
    if not chunks:
        raise ValueError("cannot convert an empty chunk list to Documents")

    return [
        Document(page_content=chunk.content, metadata=chunk.to_document_metadata())
        for chunk in chunks
    ]


def get_vectorstore(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Get a handle to a (possibly empty, possibly pre-populated) Chroma
    collection. Creates the collection on first use if it doesn't exist
    yet — Chroma handles that transparently.

    Args:
        collection_name: Name of the Chroma collection (use CODE_COLLECTION
            or DOCS_COLLECTION for the two standard collections).
        persist_directory: Directory Chroma persists to on disk.

    Returns:
        A Chroma vector store instance bound to that collection.
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_directory),
    )


def add_code_chunks(
    chunks: list[CodeChunk],
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> list[str]:
    """
    Embed and add CodeChunk objects to the code collection.

    Args:
        chunks: CodeChunk objects, e.g. from code_chunker.chunk_python_directory.
        persist_directory: Directory Chroma persists to on disk.

    Returns:
        List of Chroma-assigned document IDs for the added chunks.
    """
    store = get_vectorstore(CODE_COLLECTION, persist_directory)
    documents = _chunks_to_documents(chunks)
    return store.add_documents(documents)


def add_doc_chunks(
    chunks: list[DocChunk],
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> list[str]:
    """
    Embed and add DocChunk objects to the docs collection.

    Args:
        chunks: DocChunk objects, e.g. from doc_chunker.chunk_markdown_directory.
        persist_directory: Directory Chroma persists to on disk.

    Returns:
        List of Chroma-assigned document IDs for the added chunks.
    """
    store = get_vectorstore(DOCS_COLLECTION, persist_directory)
    documents = _chunks_to_documents(chunks)
    return store.add_documents(documents)


def collection_count(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> int:
    """
    Return how many documents are currently stored in a collection —
    useful for sanity-checking ingestion runs without a full query.

    Args:
        collection_name: CODE_COLLECTION or DOCS_COLLECTION.
        persist_directory: Directory Chroma persists to on disk.

    Returns:
        Number of documents in the collection.
    """
    store = get_vectorstore(collection_name, persist_directory)
    return store._collection.count()


def reset_collection(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> None:
    """
    Delete all documents in a collection. Useful for re-running ingestion
    from scratch during development without manually deleting the
    data/chroma_db directory (which can leave orphaned collections).

    Args:
        collection_name: CODE_COLLECTION or DOCS_COLLECTION.
        persist_directory: Directory Chroma persists to on disk.
    """
    store = get_vectorstore(collection_name, persist_directory)
    existing_ids = store.get()["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)


if __name__ == "__main__":
    # Manual sanity check: ingest the sample repo's code + docs chunks,
    # confirm both collections are populated.
    from repomind.ingestion.code_chunker import chunk_python_directory
    from repomind.ingestion.doc_chunker import chunk_markdown_directory

    reset_collection(CODE_COLLECTION)
    reset_collection(DOCS_COLLECTION)

    code_chunks = chunk_python_directory("sample_repo/src")
    doc_chunks = chunk_markdown_directory("sample_repo/docs")

    add_code_chunks(code_chunks)
    add_doc_chunks(doc_chunks)

    print(f"Code collection count: {collection_count(CODE_COLLECTION)}")
    print(f"Docs collection count: {collection_count(DOCS_COLLECTION)}")

    # Quick retrieval smoke test against the code collection.
    store = get_vectorstore(CODE_COLLECTION)
    results = store.similarity_search("How are passwords hashed?", k=2)
    print("\nTop result for 'How are passwords hashed?':")
    print(f"  {results[0].metadata['name']} ({results[0].metadata['source']})")