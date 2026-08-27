"""
doc_chunker.py — Header-based chunking for markdown documentation.

Splits a .md file into one chunk per section, using markdown headers as
the natural semantic boundary rather than a fixed character window. This
mirrors code_chunker.py's philosophy: preserve the author's own structural
units (functions/classes for code, sections for docs) instead of cutting
at arbitrary character counts.

We split on H1 and H2 (`#`, `##`) headers. This repo's docs use a single
H1 for the doc title and H2 for each topic section (e.g. "## Password
Hashing"), which is the right granularity for retrieval — fine enough
that a query about one function's behavior pulls in mostly-relevant text,
coarse enough that related sentences within a section stay together.
"""

from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter


@dataclass
class DocChunk:
    """A single retrievable unit of markdown documentation."""

    content: str                 # section body text (headers stripped by the splitter)
    file_path: str               # path relative to the repo root
    chunk_type: str = "doc_section"
    title: str = ""              # nearest H1, e.g. "Authentication"
    section: str = ""            # nearest H2, e.g. "Password Reset"
    metadata: dict = field(default_factory=dict)

    def to_document_metadata(self) -> dict:
        """Flatten metadata into a dict suitable for a LangChain Document."""
        return {
            "source": self.file_path,
            "chunk_type": self.chunk_type,
            "title": self.title,
            "section": self.section,
            **self.metadata,
        }


_HEADERS_TO_SPLIT_ON = [
    ("#", "title"),
    ("##", "section"),
]


def chunk_markdown_file(file_path: str | Path, repo_root: str | Path | None = None) -> list[DocChunk]:
    """
    Parse a single markdown file and return one DocChunk per H2 section
    (with the H1 title carried along as metadata on every chunk).

    Args:
        file_path: Path to the .md file to chunk.
        repo_root: If given, chunk.file_path is stored relative to this
            root; otherwise the path is stored as given.

    Returns:
        A list of DocChunk objects. Empty list if the file has no content
        after stripping whitespace.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if repo_root is not None:
        try:
            display_path = str(file_path.resolve().relative_to(Path(repo_root).resolve()))
        except ValueError:
            display_path = str(file_path)
    else:
        display_path = str(file_path)

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    split_docs = splitter.split_text(text)

    chunks: list[DocChunk] = []
    for doc in split_docs:
        content = doc.page_content.strip()
        if not content:
            # A header with no body text under it before the next header —
            # skip rather than emit an empty chunk.
            continue

        title = doc.metadata.get("title", "")
        section = doc.metadata.get("section", "")

        chunks.append(
            DocChunk(
                content=content,
                file_path=display_path,
                title=title,
                section=section,
            )
        )

    return chunks


def chunk_markdown_directory(
    directory: str | Path, repo_root: str | Path | None = None
) -> list[DocChunk]:
    """
    Recursively chunk every .md file under a directory.

    Args:
        directory: Root directory to walk for .md files.
        repo_root: Passed through to chunk_markdown_file for relative
            paths; defaults to `directory` itself if not given.

    Returns:
        A flat list of DocChunk objects across all files.
    """
    directory = Path(directory)
    root = Path(repo_root) if repo_root is not None else directory

    all_chunks: list[DocChunk] = []
    for md_file in sorted(directory.rglob("*.md")):
        all_chunks.extend(chunk_markdown_file(md_file, repo_root=root))

    return all_chunks


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_repo/docs"
    results = chunk_markdown_directory(target)
    for c in results:
        preview = c.content.replace("\n", " ")[:60]
        print(f"[{c.file_path}] {c.title} > {c.section}: {preview}...")
    print(f"\nTotal chunks: {len(results)}")