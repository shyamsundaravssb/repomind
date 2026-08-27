"""
ingest.py — Orchestrates the full ingestion pipeline: load -> chunk ->
embed -> store, for both code and docs sources.

This is the single entry point Week 1's scripts/run_ingestion.py calls.
It doesn't do any chunking or embedding logic itself — that all lives in
code_chunker.py / doc_chunker.py / embeddings.py / vectorstore.py — this
module just sequences those pieces and reports what happened, so there's
one place to look for "what does a full ingestion run actually do."
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from repomind.ingestion.code_chunker import chunk_python_directory
from repomind.ingestion.doc_chunker import chunk_markdown_directory
from repomind.ingestion.vectorstore import (
    CODE_COLLECTION,
    DOCS_COLLECTION,
    add_code_chunks,
    add_doc_chunks,
    collection_count,
    reset_collection,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Summary of a single ingestion run, for logging/CLI display."""

    code_dir: str
    docs_dir: str
    code_chunks_found: int
    doc_chunks_found: int
    code_files_skipped: int
    code_chunks_stored: int
    doc_chunks_stored: int
    elapsed_seconds: float

    def __str__(self) -> str:
        lines = [
            f"Ingestion run: {self.code_dir}  +  {self.docs_dir}",
            f"  Code chunks found : {self.code_chunks_found}"
            + (f"  ({self.code_files_skipped} file(s) failed to parse, skipped)"
               if self.code_files_skipped else ""),
            f"  Doc chunks found  : {self.doc_chunks_found}",
            f"  Code chunks stored: {self.code_chunks_stored}",
            f"  Doc chunks stored : {self.doc_chunks_stored}",
            f"  Elapsed           : {self.elapsed_seconds:.2f}s",
        ]
        return "\n".join(lines)


def _count_parse_failures(code_dir: str | Path) -> int:
    """
    code_chunker.chunk_python_file returns [] on a syntax error rather than
    raising, so a failed file is otherwise silently invisible in the chunk
    count. This does a lightweight second pass purely to report how many
    .py files produced zero chunks, so failures aren't hidden in the
    ingestion report.

    Args:
        code_dir: Directory that was chunked.

    Returns:
        Count of .py files that yielded no chunks at all (parse failures
        or genuinely empty files — both worth a human glance).
    """
    from repomind.ingestion.code_chunker import chunk_python_file

    skipped = 0
    for py_file in Path(code_dir).rglob("*.py"):
        if not chunk_python_file(py_file, repo_root=code_dir):
            skipped += 1
    return skipped


def run_ingestion(
    code_dir: str | Path,
    docs_dir: str | Path,
    persist_directory: str | Path | None = None,
    reset: bool = True,
) -> IngestionReport:
    """
    Run the full ingestion pipeline over a code directory and a docs
    directory: chunk both, embed, and store into their respective Chroma
    collections.

    Args:
        code_dir: Root directory of Python source to chunk (e.g.
            "sample_repo/src", later swapped for a real repo's src path).
        docs_dir: Root directory of markdown docs to chunk (e.g.
            "sample_repo/docs").
        persist_directory: Where Chroma persists to disk. Defaults to
            vectorstore.DEFAULT_PERSIST_DIR if not given.
        reset: If True (default), clears both collections before ingesting
            — appropriate for dev iteration where re-running should not
            accumulate duplicate chunks. Set False for incremental
            ingestion once that's actually needed (not yet implemented
            beyond this flag — true incremental/diff-based ingestion is
            out of scope for Week 1).

    Returns:
        An IngestionReport summarizing what was chunked and stored.

    Raises:
        FileNotFoundError: If code_dir or docs_dir doesn't exist.
    """
    code_dir = Path(code_dir)
    docs_dir = Path(docs_dir)

    if not code_dir.exists():
        raise FileNotFoundError(f"code_dir does not exist: {code_dir}")
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir does not exist: {docs_dir}")

    start = time.monotonic()

    kwargs = {}
    if persist_directory is not None:
        kwargs["persist_directory"] = persist_directory

    if reset:
        reset_collection(CODE_COLLECTION, **kwargs)
        reset_collection(DOCS_COLLECTION, **kwargs)

    logger.info("Chunking code directory: %s", code_dir)
    code_chunks = chunk_python_directory(code_dir)
    code_files_skipped = _count_parse_failures(code_dir)
    if code_files_skipped:
        logger.warning(
            "%d Python file(s) under %s produced no chunks (parse failure "
            "or empty file) — check these manually.",
            code_files_skipped, code_dir,
        )

    logger.info("Chunking docs directory: %s", docs_dir)
    doc_chunks = chunk_markdown_directory(docs_dir)

    code_chunks_stored = 0
    if code_chunks:
        add_code_chunks(code_chunks, **kwargs)
        code_chunks_stored = len(code_chunks)
    else:
        logger.warning("No code chunks found under %s — nothing added to %s.",
                        code_dir, CODE_COLLECTION)

    doc_chunks_stored = 0
    if doc_chunks:
        add_doc_chunks(doc_chunks, **kwargs)
        doc_chunks_stored = len(doc_chunks)
    else:
        logger.warning("No doc chunks found under %s — nothing added to %s.",
                        docs_dir, DOCS_COLLECTION)

    elapsed = time.monotonic() - start

    report = IngestionReport(
        code_dir=str(code_dir),
        docs_dir=str(docs_dir),
        code_chunks_found=len(code_chunks),
        doc_chunks_found=len(doc_chunks),
        code_files_skipped=code_files_skipped,
        code_chunks_stored=code_chunks_stored,
        doc_chunks_stored=doc_chunks_stored,
        elapsed_seconds=elapsed,
    )

    # Sanity check: what we stored should match what Chroma reports.
    # A mismatch here would indicate a bug in vectorstore.py's add_* path
    # (e.g. silent dedup, partial write) worth investigating immediately
    # rather than discovering later via bad retrieval results.
    actual_code_count = collection_count(CODE_COLLECTION, **kwargs)
    actual_docs_count = collection_count(DOCS_COLLECTION, **kwargs)
    if reset and actual_code_count != code_chunks_stored:
        logger.warning(
            "Code collection count mismatch: expected %d, Chroma reports %d",
            code_chunks_stored, actual_code_count,
        )
    if reset and actual_docs_count != doc_chunks_stored:
        logger.warning(
            "Docs collection count mismatch: expected %d, Chroma reports %d",
            doc_chunks_stored, actual_docs_count,
        )

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run_ingestion(
        code_dir="sample_repo/src",
        docs_dir="sample_repo/docs",
    )
    print()
    print(report)