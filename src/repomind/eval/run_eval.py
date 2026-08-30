"""
run_eval.py — Runs the RepoMind graph against the repomind-eval-v1 LangSmith
Dataset and scores it with the evaluators in evaluators.py, producing a
tracked LangSmith experiment.

This is the Week 3 "run baseline experiment" deliverable: after this runs,
the LangSmith UI's Datasets & Experiments view shows per-question and
aggregate scores for grounding_correctness, refusal_correctness, and the
three RAGAS metrics — the numbers the project's README will eventually
cite (spec Section 7's "I improved faithfulness from X to Y" claim needs
exactly this kind of run as its baseline).

Split into two entry points (run_correctness_eval / run_ragas_eval) rather
than one combined run, plus inter-example pacing — added after the first
full run exhausted Groq's free-tier daily token quota (TPD) partway
through (14/21 examples completed; see DECISIONS.md). RAGAS's three
metrics each issue their own separate LLM calls per question on top of
the graph's own calls, making the combined run far more token-hungry than
the graph alone. Splitting means grounding/refusal correctness (cheap) can
always be re-run to get clean numbers even if RAGAS (expensive) blows the
daily budget.
"""

import logging
import time
import uuid

from dotenv import load_dotenv
from langsmith import evaluate

from repomind.eval.evaluators import (
    grounding_correctness,
    no_hallucination,
    ragas_context_precision,
    ragas_context_recall,
    ragas_faithfulness,
    refusal_correctness,
)
from repomind.graph.build import ask

load_dotenv()
logger = logging.getLogger(__name__)

DATASET_NAME = "repomind-eval-v1"

# Spacing between examples within a single evaluate() run, to reduce burst
# 429s against Groq's per-minute rate limit. Does not help with the daily
# token cap (TPD) — that requires waiting for the quota window to reset,
# or reducing how many expensive (RAGAS) calls are made per run.
PACING_DELAY_SECONDS = 10


def graph_target(inputs: dict) -> dict:
    """
    Target function LangSmith's evaluate() calls once per dataset example.

    Each call gets a fresh, unique thread_id — reusing one thread across
    eval examples would let SqliteSaver checkpoint state (e.g. retry_count
    from a prior question) leak between unrelated questions, which would
    silently corrupt eval results.

    A fixed delay after each call throttles the pace of requests against
    Groq across the whole eval run — see PACING_DELAY_SECONDS.

    Args:
        inputs: The example's `inputs` dict, i.e. {"question": ...}.

    Returns:
        The full graph state dict from ask() — evaluators read whichever
        fields they need (final_answer, refused, retrieved_chunks, etc.)
        directly off this.
    """
    thread_id = f"eval-{uuid.uuid4()}"
    result = ask(inputs["question"], thread_id=thread_id)
    time.sleep(PACING_DELAY_SECONDS)
    return result


def run_correctness_eval(dataset_name: str = DATASET_NAME, experiment_prefix: str = "repomind-correctness"):
    """
    Cheap eval pass: graph_target scored by the deterministic
    grounding_correctness/refusal_correctness evaluators plus the
    semantic no_hallucination evaluator — no RAGAS calls.

    no_hallucination adds one extra LLM call per adversarial example
    only (it's a no-op for the 18 non-adversarial questions), so this
    pass stays cheap relative to the RAGAS pass.

    Args:
        dataset_name: LangSmith Dataset to evaluate against.
        experiment_prefix: Prefix for the experiment name shown in the
            LangSmith UI.

    Returns:
        The LangSmith experiment results object.
    """
    logger.info("Starting correctness-only evaluation run against dataset=%r", dataset_name)

    results = evaluate(
        graph_target,
        data=dataset_name,
        evaluators=[grounding_correctness, refusal_correctness, no_hallucination],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
        metadata={"stage": "week3-baseline", "graph_version": "week2-critique-retry", "pass": "correctness"},
    )

    logger.info("Correctness evaluation run complete.")
    return results

def run_ragas_eval(dataset_name: str = DATASET_NAME, experiment_prefix: str = "repomind-ragas"):
    """
    Expensive eval pass: graph_target scored only by the three RAGAS
    metrics (faithfulness, context precision, context recall). Run this
    separately from run_correctness_eval — each RAGAS metric issues its
    own LLM call per question, so this pass is several times more
    token-hungry than the graph run alone, and is the pass most likely to
    hit Groq's free-tier daily token cap on a 21-question dataset.

    Args:
        dataset_name: LangSmith Dataset to evaluate against.
        experiment_prefix: Prefix for the experiment name shown in the
            LangSmith UI.

    Returns:
        The LangSmith experiment results object.
    """
    logger.info("Starting RAGAS-only evaluation run against dataset=%r", dataset_name)

    results = evaluate(
        graph_target,
        data=dataset_name,
        evaluators=[ragas_faithfulness, ragas_context_precision, ragas_context_recall],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
        metadata={"stage": "week3-baseline", "graph_version": "week2-critique-retry", "pass": "ragas"},
    )

    logger.info("RAGAS evaluation run complete.")
    return results


def run_evaluation(dataset_name: str = DATASET_NAME, experiment_prefix: str = "repomind-baseline"):
    """
    Run the full eval suite in one pass: graph_target against every
    example, scored by all five evaluators together.

    Kept for convenience/completeness, but given Groq's free-tier daily
    token cap, prefer calling run_correctness_eval() and run_ragas_eval()
    separately (see module docstring) rather than this combined function,
    unless you're confident the day's quota comfortably covers both passes
    in one run.

    Args:
        dataset_name: LangSmith Dataset to evaluate against.
        experiment_prefix: Prefix for the experiment name shown in the
            LangSmith UI — distinguishes this run from later ablation runs
            (e.g. "repomind-ablation-no-critique", "repomind-ablation-naive-chunking").

    Returns:
        The LangSmith experiment results object.
    """
    logger.info("Starting combined evaluation run against dataset=%r", dataset_name)

    results = evaluate(
        graph_target,
        data=dataset_name,
        evaluators=[
            grounding_correctness,
            refusal_correctness,
            ragas_faithfulness,
            ragas_context_precision,
            ragas_context_recall,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
        metadata={"stage": "week3-baseline", "graph_version": "week2-critique-retry", "pass": "combined"},
    )

    logger.info("Combined evaluation run complete.")
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Usage: uv run python -m repomind.eval.run_eval [correctness|ragas|combined]
    # Defaults to correctness (cheapest, safest to re-run) if no argument given.
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"

    if mode == "correctness":
        run_correctness_eval()
    elif mode == "ragas":
        run_ragas_eval()
    elif mode == "combined":
        run_evaluation()
    else:
        print(f"Unknown mode: {mode!r}. Use 'correctness', 'ragas', or 'combined'.")
        sys.exit(1)