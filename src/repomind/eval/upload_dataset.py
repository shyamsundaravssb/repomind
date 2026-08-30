"""
upload_dataset.py — Uploads eval_dataset.json to LangSmith as a versioned
Dataset, so eval runs are tracked against a stable, inspectable dataset
rather than a local file re-read on every run.

Idempotent by design: re-running this script does not create duplicate
examples. If the named dataset already exists, existing examples are
diffed against the local JSON by `id` (stored in each example's metadata)
and only new/changed entries are added or updated — this way, editing
eval_dataset.json and re-running is the normal workflow for iterating on
the eval set, not something to be avoided.
"""
import os
import json
import logging
from pathlib import Path

from langsmith import Client

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATASET_NAME = "repomind-eval-v1"
DATASET_DESCRIPTION = (
    "RepoMind Week 3 eval set: 20 hand-written Q&A pairs over sample_repo/, "
    "spanning code_lookup, how_to, conceptual, and adversarial (refusal) "
    "categories. See eval_dataset.json for source of truth; this dataset "
    "is regenerated from that file, not edited directly in the LangSmith UI."
)

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_local_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict]:
    """
    Load and validate the local eval dataset JSON file.

    Args:
        path: Path to eval_dataset.json.

    Returns:
        List of eval entry dicts.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If any entry is missing a required field, or if `id`
            values aren't unique (uniqueness is required for the diff-based
            upsert logic below to work correctly).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")

    entries = json.loads(path.read_text(encoding="utf-8"))

    required_fields = {"id", "question", "category", "expected_answer_summary",
                        "expected_grounded", "expected_refusal"}
    seen_ids = set()

    for entry in entries:
        missing = required_fields - entry.keys()
        if missing:
            raise ValueError(f"entry {entry.get('id', '<no id>')} missing fields: {missing}")
        if entry["id"] in seen_ids:
            raise ValueError(f"duplicate id in eval dataset: {entry['id']}")
        seen_ids.add(entry["id"])

    return entries


def upload_dataset(
    local_path: str | Path = DEFAULT_DATASET_PATH,
    dataset_name: str = DATASET_NAME,
) -> dict:
    """
    Create (if needed) a LangSmith Dataset and upsert examples from the
    local JSON file into it, keyed by each entry's `id`.

    Each LangSmith example is created with:
        - inputs: {"question": ...}
        - outputs: {"expected_answer_summary": ..., "expected_grounded": ...,
                     "expected_refusal": ...}
        - metadata: {"id": ..., "category": ...}

    Splitting expected fields into `outputs` (what a correct run should
    produce) vs `metadata` (bookkeeping/filtering, like category and id)
    follows LangSmith's convention so built-in and custom evaluators can
    reference `example.outputs` directly.

    Args:
        local_path: Path to eval_dataset.json.
        dataset_name: Name of the LangSmith Dataset to create/update.

    Returns:
        Summary dict: {"created": int, "updated": int, "unchanged": int,
        "dataset_id": str}.
    """
    entries = load_local_dataset(local_path)

    # Client() does not reliably pick up LANGSMITH_ENDPOINT/LANGCHAIN_ENDPOINT
    # from the environment the same way LangChain's auto-tracing does — pass
    # it explicitly to avoid silently falling back to the US endpoint, which
    # 401s for this APAC-region account (see DECISIONS.md).
    api_url = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    if not api_url or not api_key:
        raise ValueError(
            "LANGSMITH_ENDPOINT/LANGCHAIN_ENDPOINT and LANGSMITH_API_KEY/"
            "LANGCHAIN_API_KEY must be set in .env."
        )

    client = Client(api_url=api_url, api_key=api_key)
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info("Found existing dataset %r (id=%s)", dataset_name, dataset.id)
    else:
        dataset = client.create_dataset(dataset_name=dataset_name, description=DATASET_DESCRIPTION)
        logger.info("Created new dataset %r (id=%s)", dataset_name, dataset.id)

    # Build a lookup of existing examples by our own `id` (stored in metadata),
    # not LangSmith's internal example id, so this script can be re-run safely
    # after edits to eval_dataset.json.
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    existing_by_local_id = {
        ex.metadata.get("id"): ex
        for ex in existing_examples
        if ex.metadata and ex.metadata.get("id")
    }

    created = updated = unchanged = 0

    for entry in entries:
        inputs = {"question": entry["question"]}
        outputs = {
            "expected_answer_summary": entry["expected_answer_summary"],
            "expected_grounded": entry["expected_grounded"],
            "expected_refusal": entry["expected_refusal"],
        }
        metadata = {"id": entry["id"], "category": entry["category"]}

        existing = existing_by_local_id.get(entry["id"])

        if existing is None:
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                dataset_id=dataset.id,
            )
            created += 1
        else:
            # Compare against what's already stored; only issue an update
            # call if something actually changed, to avoid needlessly
            # bumping every example's version on every script run.
            if existing.inputs != inputs or existing.outputs != outputs or existing.metadata != metadata:
                client.update_example(
                    example_id=existing.id,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
                updated += 1
            else:
                unchanged += 1

    summary = {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "dataset_id": str(dataset.id),
    }
    logger.info("Upload complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = upload_dataset()
    print(f"\nDataset: {DATASET_NAME}")
    print(f"  Created:   {result['created']}")
    print(f"  Updated:   {result['updated']}")
    print(f"  Unchanged: {result['unchanged']}")
    print(f"  Dataset ID: {result['dataset_id']}")