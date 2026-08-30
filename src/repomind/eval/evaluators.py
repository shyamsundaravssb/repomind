"""
evaluators.py — Evaluator functions for the RepoMind eval pipeline.

Two families of evaluators:
1. Custom, deterministic evaluators (grounding_correctness, refusal_correctness)
   that check the graph's own verdict (is_grounded/refused) against the
   eval dataset's expected labels — these measure whether the critique
   node's judgment matches ground truth, independent of answer quality.
2. RAGAS metrics (faithfulness, context_precision, context_recall) that
   measure answer/retrieval quality directly, independent of what the
   critique node claimed about itself — this is the check on whether the
   critique node's self-reported groundedness is actually trustworthy.

Both families matter for different reasons: (1) tells you if the refusal
mechanism works; (2) tells you if "grounded" answers are actually faithful
to the retrieved context, which is the deeper claim the whole project
rests on.
"""



import sys
import types

# --- Compatibility shim (forced workaround, not a design choice) -----------
# ragas==0.4.3 unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai at module load time, purely for
# an isinstance() check deep in ragas/llms/base.py. That submodule was
# removed from langchain_community (Google integrations split into a
# separate package upstream), so the bare import crashes before ragas even
# loads — regardless of whether Vertex AI is used anywhere in this project.
# Since the class is never instantiated unless you actually use Vertex,
# registering a dummy stand-in class under the expected module path is
# sufficient to satisfy the import without pulling in Google Cloud
# dependencies this project doesn't use. Documented in DECISIONS.md.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:
        """Dummy stand-in — never instantiated, only used by ragas for isinstance checks."""
        pass

    _vertexai_shim.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim
# --- end shim ---------------------------------------------------------------


import logging

from langsmith.schemas import Example, Run
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_core.messages import HumanMessage, SystemMessage

from repomind.generation.llm import get_llm

logger = logging.getLogger(__name__)

# RAGAS metrics need an LLM to judge faithfulness/precision/recall against
# the retrieved context — reuse the same Groq model already configured for
# the rest of the project rather than introducing a second LLM dependency.
_ragas_llm = LangchainLLMWrapper(get_llm())


# --- Custom deterministic evaluators ---------------------------------------------------------------

def grounding_correctness(run: Run, example: Example) -> dict:
    """
    Checks whether the graph's own is_grounded/refused verdict matches the
    eval dataset's expected_grounded label — measures whether the
    critique node's judgment is calibrated, not whether the final answer
    text is any good.

    Args:
        run: The LangSmith Run for this example (run.outputs is the dict
            returned by ask()/the compiled graph).
        example: The LangSmith Example, with example.outputs holding
            expected_grounded/expected_refusal from the eval dataset.

    Returns:
        {"key": "grounding_correctness", "score": 1 or 0, "comment": str}
    """
    expected_grounded = example.outputs.get("expected_grounded")
    actual_refused = run.outputs.get("refused", None)

    if actual_refused is None:
        return {"key": "grounding_correctness", "score": 0, "comment": "run produced no 'refused' field"}

    # A non-refused answer implies the graph judged itself grounded.
    actual_grounded = not actual_refused
    score = int(actual_grounded == expected_grounded)

    comment = (
        f"expected_grounded={expected_grounded}, actual (not refused)={actual_grounded}"
    )
    return {"key": "grounding_correctness", "score": score, "comment": comment}


def refusal_correctness(run: Run, example: Example) -> dict:
    """
    Checks whether the graph refused exactly when it was expected to —
    the direct measure of the adversarial test cases' pass/fail state.

    Args:
        run: The LangSmith Run for this example.
        example: The LangSmith Example, with expected_refusal in outputs.

    Returns:
        {"key": "refusal_correctness", "score": 1 or 0, "comment": str}
    """
    expected_refusal = example.outputs.get("expected_refusal")
    actual_refused = run.outputs.get("refused", None)

    if actual_refused is None:
        return {"key": "refusal_correctness", "score": 0, "comment": "run produced no 'refused' field"}

    score = int(actual_refused == expected_refusal)
    comment = f"expected_refusal={expected_refusal}, actual_refused={actual_refused}"
    return {"key": "refusal_correctness", "score": score, "comment": comment}



# --- Semantic no-hallucination evaluator ---------------------------------------------------------------

NO_HALLUCINATION_JUDGE_PROMPT = """You are judging whether an AI system's answer avoided hallucination, for a question where the correct information may genuinely be absent from the system's available context.

The system was allowed to reach a "non-hallucinating" outcome in two different ways, both acceptable:
1. Explicitly refusing to answer (stating it couldn't find a grounded answer).
2. Directly answering, but honestly stating that the requested information/feature does not appear to exist in the provided code/docs — without presenting any speculative or fabricated specifics as if they were confirmed facts.

The system FAILS this check if the answer confidently presents a specific fact (a number, a function's existence, a behavior) as true/implemented, when that fact is only present in documentation with no corresponding code support, and the answer does not clearly flag that gap.

Question: {question}

System's final answer:
{answer}

Respond in exactly this format, nothing else:
NO_HALLUCINATION: yes or no
REASON: one sentence explaining your judgment.
"""


def no_hallucination(run: Run, example: Example) -> dict:
    """
    Semantic check for whether the final answer avoided hallucination,
    independent of which graph code path (refuse_node vs. an honest
    "not found" answer from synthesize_node/critique_node) was taken.

    This exists because the binary `refused` field only captures the
    retry-exhaustion path — a run that reaches is_grounded=True on the
    first pass because the draft honestly says "this isn't in the
    context" is functionally non-hallucinating but scores 0 under
    refusal_correctness, which only checks refused == expected_refusal.
    This evaluator measures the actual thing of interest (did it fabricate
    a fact) rather than the specific mechanism used to avoid doing so.

    Only meaningful for adversarial/expected_refusal=True examples — for
    a normal grounded question expecting a real answer, this check isn't
    relevant, since presenting a confirmed fact is the correct behavior,
    not a hallucination risk. Skipped (returns None) for non-adversarial
    examples to avoid a nonsensical score.

    Args:
        run: The LangSmith Run for this example.
        example: The LangSmith Example, with expected_refusal in outputs.

    Returns:
        {"key": "no_hallucination", "score": 1 or 0 or None, "comment": str}
    """
    expected_refusal = example.outputs.get("expected_refusal")
    if not expected_refusal:
        return {"key": "no_hallucination", "score": None, "comment": "skipped: not an adversarial example"}

    answer = run.outputs.get("final_answer", "")
    question = example.inputs.get("question", "")

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a careful, skeptical fact-checker. Follow the instructions exactly."),
        HumanMessage(content=NO_HALLUCINATION_JUDGE_PROMPT.format(question=question, answer=answer)),
    ])

    raw = response.content.strip()
    passed = raw.upper().startswith("NO_HALLUCINATION: YES")

    reason = raw
    if "REASON:" in raw:
        reason = raw.split("REASON:", 1)[1].strip()

    return {"key": "no_hallucination", "score": int(passed), "comment": reason}

# --- RAGAS metric evaluators ---------------------------------------------------------------

def _build_ragas_sample(run: Run, example: Example) -> SingleTurnSample | None:
    """
    Construct a RAGAS SingleTurnSample from a graph run's outputs.

    Returns None if the run refused — RAGAS's faithfulness/precision/recall
    metrics assume there's a substantive answer to score against context;
    scoring a refusal message against retrieved context isn't a meaningful
    faithfulness check (a refusal has no factual claims to verify). Refusal
    correctness itself is already covered by refusal_correctness() above.

    Args:
        run: The LangSmith Run.
        example: The LangSmith Example.

    Returns:
        A SingleTurnSample, or None if this run should be skipped (refused).
    """
    if run.outputs.get("refused"):
        return None

    chunks = run.outputs.get("retrieved_chunks", [])
    contexts = [c["content"] for c in chunks]

    return SingleTurnSample(
        user_input=example.inputs["question"],
        response=run.outputs.get("final_answer", ""),
        retrieved_contexts=contexts,
        reference=example.outputs.get("expected_answer_summary", ""),
    )


def ragas_faithfulness(run: Run, example: Example) -> dict:
    """
    RAGAS faithfulness: what fraction of claims in the final answer are
    inferable from the retrieved context. This is the core "is it actually
    grounded, not just self-certified as grounded" check.

    Args:
        run: The LangSmith Run.
        example: The LangSmith Example.

    Returns:
        {"key": "ragas_faithfulness", "score": float, "comment": str}
    """
    sample = _build_ragas_sample(run, example)
    if sample is None:
        return {"key": "ragas_faithfulness", "score": None, "comment": "skipped: run refused, no answer to score"}

    metric = Faithfulness(llm=_ragas_llm)
    score = metric.single_turn_score(sample)
    return {"key": "ragas_faithfulness", "score": score}


def ragas_context_precision(run: Run, example: Example) -> dict:
    """
    RAGAS context precision: of the chunks retrieved, what fraction were
    actually relevant/useful for answering the question. Low precision
    means the retriever is pulling in noise alongside signal.

    Args:
        run: The LangSmith Run.
        example: The LangSmith Example.

    Returns:
        {"key": "ragas_context_precision", "score": float, "comment": str}
    """
    sample = _build_ragas_sample(run, example)
    if sample is None:
        return {"key": "ragas_context_precision", "score": None, "comment": "skipped: run refused"}

    metric = LLMContextPrecisionWithReference(llm=_ragas_llm)
    score = metric.single_turn_score(sample)
    return {"key": "ragas_context_precision", "score": score}


def ragas_context_recall(run: Run, example: Example) -> dict:
    """
    RAGAS context recall: of the information needed to answer the question
    (per the reference summary), what fraction was actually present
    somewhere in the retrieved context. Low recall means relevant chunks
    exist but weren't retrieved.

    Args:
        run: The LangSmith Run.
        example: The LangSmith Example.

    Returns:
        {"key": "ragas_context_recall", "score": float, "comment": str}
    """
    sample = _build_ragas_sample(run, example)
    if sample is None:
        return {"key": "ragas_context_recall", "score": None, "comment": "skipped: run refused"}

    metric = LLMContextRecall(llm=_ragas_llm)
    score = metric.single_turn_score(sample)
    return {"key": "ragas_context_recall", "score": score}