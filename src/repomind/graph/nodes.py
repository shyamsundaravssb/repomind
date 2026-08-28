"""
nodes.py — Node functions for the RepoMind LangGraph pipeline.

Each node is a plain function: (GraphState) -> partial GraphState update.
No LangGraph-specific machinery lives here — that's all in build.py, which
wires these functions into a StateGraph with edges/conditionals. Keeping
that separation means each node is independently testable/callable without
constructing a graph at all.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from repomind.generation.llm import get_llm
from repomind.graph.state import GraphState, QueryType, RetrievedChunkDict
from repomind.retrieval.retriever import CODE_COLLECTION, DOCS_COLLECTION, get_vectorstore



logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2



def _format_context(chunks: list[RetrievedChunkDict]) -> str:
    """
    Format retrieved chunk dicts into a labeled context string for LLM
    prompts. Dict-based equivalent of retriever.format_context, kept
    separate so nodes.py has no dependency on the RetrievedChunk dataclass
    (which is not checkpoint-safe — see state.py's RetrievedChunkDict).

    Args:
        chunks: List of chunk dicts as stored in GraphState.

    Returns:
        A formatted string, one labeled block per chunk.
    """
    blocks = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        if chunk["source_collection"] == CODE_COLLECTION:
            label = f"[Code: {metadata.get('source')} — {metadata.get('name')}]"
        else:
            label = f"[Docs: {metadata.get('source')} — {metadata.get('section') or metadata.get('title')}]"
        blocks.append(f"{label}\n{chunk['content']}")

    return "\n\n---\n\n".join(blocks)

# --- Router ---------------------------------------------------------------

ROUTER_PROMPT = """Classify the following question about a codebase into exactly one category:

- code_lookup: asking what a specific function/class/constant does, its signature, or its implementation details.
- how_to: asking how to accomplish a task or use a feature (procedural).
- conceptual: asking about overall architecture, design, or how a system works at a higher level than one function.

Respond with ONLY the category name, nothing else: code_lookup, how_to, or conceptual.

Question: {question}"""


def router_node(state: GraphState) -> dict:
    """
    Classify the incoming question into one of three query types, used to
    weight retrieval toward code vs. docs in the retrieve node.

    Note: the spec's fourth category, issue_history, is intentionally not
    classified into yet — there's no issues/PR retriever built (Week 2
    stretch goal), so routing into a category with no corresponding source
    would be a dead end. Revisit once that retriever exists.

    Args:
        state: Current graph state; reads state["question"].

    Returns:
        Partial state update: {"query_type": ..., "retry_count": 0,
        "max_retries": ...} — retry counters initialized here since this
        is the first node in the graph.
    """
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a query classifier. Follow the instructions exactly."),
        HumanMessage(content=ROUTER_PROMPT.format(question=state["question"])),
    ])

    raw = response.content.strip().lower()
    valid_types: set[QueryType] = {"code_lookup", "how_to", "conceptual"}
    query_type: QueryType = raw if raw in valid_types else "conceptual"  # safe default on unexpected output

    if raw not in valid_types:
        logger.warning("Router returned unrecognized category %r, defaulting to 'conceptual'", raw)

    logger.info("Router classified question as: %s", query_type)

    return {
        "query_type": query_type,
        "retry_count": 0,
        "max_retries": state.get("max_retries", DEFAULT_MAX_RETRIES),
    }


# --- Retrieve ---------------------------------------------------------------

# Per query_type, how many chunks to pull from each collection. code_lookup
# skews toward code; how_to and conceptual skew toward docs, since "how do
# I..." and architecture questions are usually answered in prose, with code
# as supporting evidence rather than the primary source.
_RETRIEVAL_WEIGHTS: dict[QueryType, dict[str, int]] = {
    "code_lookup": {"code_k": 5, "docs_k": 2},
    "how_to": {"code_k": 2, "docs_k": 4},
    "conceptual": {"code_k": 2, "docs_k": 4},
}


def retrieve_node(state: GraphState) -> dict:
    """
    Retrieve chunks from both collections, weighted by query_type — this
    is the Week 2 "router + multi-retriever" path, distinct from
    retrieval/retriever.py's flat merged baseline used in Week 1.

    On a reformulation retry, retrieves using state["question"] as it
    currently stands (reformulate_node updates state["question"] before
    looping back here), not the original question.

    Args:
        state: Current graph state; reads state["question"], state["query_type"].

    Returns:
        Partial state update: {"retrieved_chunks": [...]}.
    """
    query_type = state.get("query_type", "conceptual")
    weights = _RETRIEVAL_WEIGHTS[query_type]

    code_store = get_vectorstore(CODE_COLLECTION)
    docs_store = get_vectorstore(DOCS_COLLECTION)

    code_results = code_store.similarity_search_with_score(state["question"], k=weights["code_k"])
    docs_results = docs_store.similarity_search_with_score(state["question"], k=weights["docs_k"])

    chunks: list[RetrievedChunkDict] = []
    for doc, score in code_results:
        chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
            "source_collection": CODE_COLLECTION,
        })
    for doc, score in docs_results:
        chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
            "source_collection": DOCS_COLLECTION,
        })

    chunks.sort(key=lambda c: c["score"])

    logger.info("Retrieved %d chunks (%d code, %d docs) for query_type=%s",
                len(chunks), len(code_results), len(docs_results), query_type)

    return {"retrieved_chunks": chunks}


# --- Synthesize ---------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are a codebase Q&A assistant. Answer the user's question using ONLY the context provided below — do not use any outside knowledge about the library or framework in question.

Rules:
- If the context fully answers the question, answer clearly and cite which file/function or doc section you used.
- If the context is documentation describing something (e.g. a function) that does NOT actually appear in the code context provided, say so explicitly — do not assume it exists just because it's documented.
- If the context does not contain enough information to answer the question, say clearly that the codebase/docs provided don't contain this information. Do not guess or fabricate function names, parameters, or behavior.
"""


def synthesize_node(state: GraphState) -> dict:
    """
    Draft an answer from the retrieved chunks. This is prompt-level
    grounding only (same instruction as Week 1's baseline) — the actual
    verification of whether the draft honored these instructions happens
    in critique_node, not here.

    Args:
        state: Current graph state; reads state["question"], state["retrieved_chunks"].

    Returns:
        Partial state update: {"draft_answer": ...}.
    """
    context = _format_context(state["retrieved_chunks"])
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
    ])

    return {"draft_answer": response.content}


# --- Critique ---------------------------------------------------------------

CRITIQUE_SYSTEM_PROMPT = """You are a strict fact-checker reviewing a draft answer against the exact context it was supposedly generated from.

Your job: determine whether the draft answer's claims are grounded in the CODE portion of the context, not just the documentation portion. This distinction matters: documentation can describe features, functions, or behavior that were never actually implemented. A draft that accurately quotes documentation is NOT automatically grounded — it is only grounded if the underlying feature/function it describes is also confirmed by the CODE portion of the context.

Flag as NOT grounded if ANY of the following apply:
- The draft describes a function, class, or specific behavior that appears ONLY in the documentation portion of the context, with no corresponding function/class in the CODE portion.
- The draft states a specific numeric detail (rate limits, timeouts, sizes, counts) that is sourced from documentation but has no corresponding code enforcing or implementing it in the CODE portion.
- The CODE portion of the context contains nothing relevant to the question at all — this itself is a signal the feature may not exist, even if documentation describes it confidently.

Do NOT assume something exists just because documentation describes it in detail or the draft quotes it accurately — accurate quoting of unverified documentation is still NOT grounded.

Respond in exactly this format, nothing else:
GROUNDED: yes or no
NOTES: one or two sentences explaining your judgment, and if not grounded, specifically what code-side corroboration was missing.
"""


def _code_names_in_context(chunks: list[RetrievedChunkDict]) -> set[str]:
    """
    Extract the set of function/class names actually present in the CODE
    portion of retrieved context.

    Args:
        chunks: Retrieved chunk dicts (mixed code + docs).

    Returns:
        Lowercased set of code chunk names (function/class names) present.
    """
    return {
        c["metadata"].get("name", "").lower()
        for c in chunks
        if c["source_collection"] == CODE_COLLECTION and c["metadata"].get("name")
    }

def critique_node(state: GraphState) -> dict:
    """
    Check the draft answer's claims against the retrieved chunks — this is
    the programmatic grounding check that Week 1's system-prompt-only
    approach lacked, and it's specifically what should catch cases like
    the password-reset adversarial test (docs describe a function that
    doesn't exist in the code context).

    Combines an LLM-judged critique with a cheap programmatic backstop:
    if the retrieved CODE chunks are essentially unrelated to the doc
    chunks the draft leans on (no shared/relevant code names at all), that
    alone is treated as ungrounded regardless of the LLM's verdict — this
    guards against the specific failure mode observed in testing, where
    the critique LLM judged a docs-only claim "grounded" because it
    accurately quoted the docs without checking for code corroboration.

    Args:
        state: Current graph state; reads state["draft_answer"], state["retrieved_chunks"].

    Returns:
        Partial state update: {"is_grounded": bool, "critique_notes": str}.
    """
    chunks = state["retrieved_chunks"]
    context = _format_context(chunks)
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nDraft answer:\n{state['draft_answer']}"),
    ])

    raw = response.content.strip()
    llm_grounded = raw.upper().startswith("GROUNDED: YES")

    notes = raw
    if "NOTES:" in raw:
        notes = raw.split("NOTES:", 1)[1].strip()

    code_names = _code_names_in_context(chunks)
    has_code_support = len(code_names) > 0

    is_grounded = llm_grounded and has_code_support

    if llm_grounded and not has_code_support:
        notes = (
            "Overridden by programmatic check: no code chunks were retrieved "
            "to corroborate this answer, despite the LLM critique judging it "
            "grounded. " + notes
        )

    logger.info("Critique verdict: llm_grounded=%s, has_code_support=%s, final=%s",
                llm_grounded, has_code_support, is_grounded)

    return {"is_grounded": is_grounded, "critique_notes": notes}

# --- Reformulate ---------------------------------------------------------------

REFORMULATE_PROMPT = """The following question was answered from retrieved context, but the answer was judged NOT sufficiently grounded in that context — likely because retrieval didn't surface the right chunks.

Original question: {question}
Why the answer was ungrounded: {critique_notes}

Rewrite the question to be more likely to retrieve the right context — for example, using more specific terminology, or rephrasing a vague question more concretely. Respond with ONLY the rewritten question, nothing else."""


def reformulate_node(state: GraphState) -> dict:
    """
    Rewrite the question for a retry, and increment retry_count. Runs only
    when critique_node judged the draft ungrounded and retries remain.

    Args:
        state: Current graph state; reads state["question"], state["critique_notes"], state["retry_count"].

    Returns:
        Partial state update: {"question": ..., "retry_count": ...}.
    """
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You rewrite questions to improve retrieval, nothing else."),
        HumanMessage(content=REFORMULATE_PROMPT.format(
            question=state["question"],
            critique_notes=state.get("critique_notes", ""),
        )),
    ])

    new_question = response.content.strip()
    new_retry_count = state.get("retry_count", 0) + 1

    logger.info("Reformulated question (retry %d): %s", new_retry_count, new_question)

    return {"question": new_question, "retry_count": new_retry_count}


# --- Finalize ---------------------------------------------------------------

def finalize_node(state: GraphState) -> dict:
    """
    Terminal node: sets final_answer from draft_answer when grounded.

    Args:
        state: Current graph state; reads state["draft_answer"].

    Returns:
        Partial state update: {"final_answer": ..., "refused": False}.
    """
    return {"final_answer": state["draft_answer"], "refused": False}


def refuse_node(state: GraphState) -> dict:
    """
    Terminal node reached when max_retries is exhausted without reaching
    groundedness — returns an explicit refusal rather than the last
    (still-ungrounded) draft, per the spec's requirement to refuse rather
    than hallucinate when the repo genuinely doesn't contain the answer.

    Args:
        state: Current graph state; reads state["critique_notes"].

    Returns:
        Partial state update: {"final_answer": ..., "refused": True}.
    """
    notes = state.get("critique_notes", "")
    final = (
        "I wasn't able to find a well-grounded answer to this question in the "
        "provided codebase and documentation after retrying the search. "
        f"({notes})" if notes else
        "I wasn't able to find a well-grounded answer to this question in the "
        "provided codebase and documentation after retrying the search."
    )
    return {"final_answer": final, "refused": True}