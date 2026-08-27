"""
run_ingestion.py — CLI entrypoint for the ingestion pipeline.

Thin wrapper around repomind.ingestion.ingest.run_ingestion — argument
parsing only, no pipeline logic here.

Usage:
    uv run python scripts/run_ingestion.py --code sample_repo/src --docs sample_repo/docs
"""

import argparse
import logging

from repomind.ingestion.ingest import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RepoMind ingestion pipeline.")
    parser.add_argument("--code", required=True, help="Path to the code directory to ingest.")
    parser.add_argument("--docs", required=True, help="Path to the docs directory to ingest.")
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Chroma persist directory (defaults to data/chroma_db).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip clearing existing collections before ingesting (incremental-ish; "
             "duplicates possible — see run_ingestion's reset docstring).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run_ingestion(
        code_dir=args.code,
        docs_dir=args.docs,
        persist_directory=args.persist_dir,
        reset=not args.no_reset,
    )
    print()
    print(report)


if __name__ == "__main__":
    main()