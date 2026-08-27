"""
code_chunker.py — AST-aware chunking for Python source files.

Rather than splitting code by a fixed character/token window (which can
sever a function or class mid-body), this walks the module's AST and emits
one chunk per top-level function, async function, and class — each chunk
containing the *complete* source of that unit. Methods inside a class are
kept as part of the class chunk rather than split out individually; this
preserves the class as the semantic unit a developer would actually want
retrieved (you rarely want just one method's body without its siblings/
attributes for context).

Trade-off: a very large class becomes one large chunk, which can hurt
retrieval precision if only one method is relevant to the query. This is
a known limitation to surface in the ablation writeup — a finer-grained
"class chunk + method sub-chunks" strategy is a plausible follow-up
experiment, not implemented here to keep Week 1 scope bounded.

Module-level code that isn't inside a function/class/docstring (e.g. plain
statements, constants) is captured separately as a single "module-level"
chunk per file, so it isn't silently dropped.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunk:
    """A single retrievable unit of source code."""

    content: str                     # exact source text for this unit
    file_path: str                   # path relative to the repo root
    chunk_type: str                  # "function" | "async_function" | "class" | "module_level"
    name: str                        # function/class name, or "<module>" for module-level
    start_line: int
    end_line: int
    docstring: str | None = None
    parent_class: str | None = None  # set if this were a method-level chunk (unused currently, reserved)
    metadata: dict = field(default_factory=dict)

    def to_document_metadata(self) -> dict:
        """Flatten metadata into a dict suitable for a LangChain Document."""
        return {
            "source": self.file_path,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring or "",
            **self.metadata,
        }


def _get_docstring(node: ast.AST) -> str | None:
    """Extract a docstring from a function/class/module AST node, if present."""
    try:
        return ast.get_docstring(node)
    except TypeError:
        return None


def _node_source(source_lines: list[str], node: ast.AST) -> tuple[str, int, int]:
    """
    Slice the exact original source text for an AST node, including any
    decorators (ast's lineno for a decorated function starts at the `def`,
    not the decorator, so we walk back over decorator_list explicitly).

    Returns (source_text, start_line, end_line), both 1-indexed inclusive.
    """
    start_line = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start_line = min(d.lineno for d in decorators)

    end_line = getattr(node, "end_lineno", None)
    if end_line is None:
        # Fallback for older Python without end_lineno; scan forward until
        # dedent below the node's column offset. Kept simple since we
        # target 3.11+ where end_lineno is always populated.
        end_line = start_line

    chunk_lines = source_lines[start_line - 1 : end_line]
    return "\n".join(chunk_lines), start_line, end_line


def chunk_python_file(file_path: str | Path, repo_root: str | Path | None = None) -> list[CodeChunk]:
    """
    Parse a single Python file and return one CodeChunk per top-level
    function/async function/class, plus one chunk for any remaining
    module-level code.

    Args:
        file_path: Path to the .py file to chunk.
        repo_root: If given, chunk.file_path is stored relative to this
            root; otherwise the path is stored as given.

    Returns:
        A list of CodeChunk objects. Empty list if the file fails to parse
        (e.g. syntax error) — callers should log this rather than crash
        the whole ingestion run over one bad file.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    if repo_root is not None:
        try:
            display_path = str(file_path.resolve().relative_to(Path(repo_root).resolve()))
        except ValueError:
            display_path = str(file_path)
    else:
        display_path = str(file_path)

    chunks: list[CodeChunk] = []
    covered_lines: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            content, start_line, end_line = _node_source(source_lines, node)
            covered_lines.update(range(start_line, end_line + 1))

            if isinstance(node, ast.ClassDef):
                chunk_type = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            else:
                chunk_type = "function"

            method_names = []
            if isinstance(node, ast.ClassDef):
                method_names = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

            chunks.append(
                CodeChunk(
                    content=content,
                    file_path=display_path,
                    chunk_type=chunk_type,
                    name=node.name,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=_get_docstring(node),
                    metadata={"methods": method_names} if method_names else {},
                )
            )

    # Anything not covered by a top-level function/class chunk (imports,
    # module docstring, constants, top-level statements) becomes one
    # module-level chunk, so ingestion doesn't silently drop it.
    leftover_numbered = [
        (i, line) for i, line in enumerate(source_lines, start=1)
        if i not in covered_lines
    ]
    # Only lines with real content count toward the reported range —
    # a leftover blank line at the top/bottom of the file shouldn't
    # widen start_line/end_line.
    nonblank_numbers = [i for i, line in leftover_numbered if line.strip()]

    if nonblank_numbers:
        leftover_content = "\n".join(line for _, line in leftover_numbered).strip()
        chunks.append(
            CodeChunk(
                content=leftover_content,
                file_path=display_path,
                chunk_type="module_level",
                name="<module>",
                start_line=min(nonblank_numbers),
                end_line=max(nonblank_numbers),
                docstring=_get_docstring(tree),
            )
        )


    return chunks


def chunk_python_directory(
    directory: str | Path, repo_root: str | Path | None = None
) -> list[CodeChunk]:
    """
    Recursively chunk every .py file under a directory.

    Args:
        directory: Root directory to walk for .py files.
        repo_root: Passed through to chunk_python_file for relative paths;
            defaults to `directory` itself if not given.

    Returns:
        A flat list of CodeChunk objects across all files. Files that fail
        to parse are skipped (see chunk_python_file).
    """
    directory = Path(directory)
    root = Path(repo_root) if repo_root is not None else directory

    all_chunks: list[CodeChunk] = []
    for py_file in sorted(directory.rglob("*.py")):
        all_chunks.extend(chunk_python_file(py_file, repo_root=root))

    return all_chunks


if __name__ == "__main__":
    # Quick manual sanity check against the sample repo.
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_repo/src"
    results = chunk_python_directory(target)
    for c in results:
        print(f"[{c.chunk_type:14s}] {c.file_path}:{c.start_line}-{c.end_line}  {c.name}")
    print(f"\nTotal chunks: {len(results)}")