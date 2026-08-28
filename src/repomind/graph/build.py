"""
build.py — Wires the node functions in nodes.py into a LangGraph
StateGraph: routing, conditional critique/retry loop, and SqliteSaver
checkpointing.

This is the only file that imports langgraph directly — nodes.py stays
framework-agnostic (plain functions), so this module's job is purely
graph topology: which node follows which, and the conditional edge that
decides retry vs. finalize vs. refuse.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from repomind.graph.nodes import (
    critique_node,
    finalize_node,
    reformulate_node,
    refuse_node,
    retrieve_node,
    router_node,
    synthesize_node,
)
from repomind.graph.state import GraphState

DEFAULT_CHECKPOINT_DB = Path("data/checkpoints.sqlite")


def _critique_router(state: GraphState) -> str:
    """
    Conditional edge function: decides where to go after critique_node.

    Returns:
        "finalize" if grounded.
        "reformulate" if not grounded and retries remain.
        "refuse" if not grounded and retries are exhausted.
    """
    if state.get("is_grounded"):
        return "finalize"

    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "reformulate"

    return "refuse"


def build_graph(checkpoint_db: str | Path = DEFAULT_CHECKPOINT_DB):
    """
    Construct and compile the RepoMind graph.

    Graph shape:
        router -> retrieve -> synthesize -> critique
                                                |
                        (conditional: grounded / retry / exhausted)
                                                |
                    finalize <--- OR ---> reformulate -> retrieve (loop)
                       |                          OR ---> refuse
                      END                                  |
                                                            END

    Args:
        checkpoint_db: Path to the SQLite file backing SqliteSaver. The
            parent directory is created if it doesn't exist. Using a
            file-backed (not in-memory) connection so state actually
            survives process restarts, per the spec's checkpointing
            requirement.

    Returns:
        A compiled LangGraph graph, invokable via .invoke(input, config).
    """
    checkpoint_db = Path(checkpoint_db)
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: LangGraph's SqliteSaver may be called from
    # a different thread than the one that opened the connection,
    # depending on how the graph is invoked (e.g. from an async/FastAPI
    # context later in Week 4). Safe here since we're not sharing this
    # connection across concurrent writers within a single process in
    # Week 2's usage pattern.
    conn = sqlite3.connect(str(checkpoint_db), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critique", critique_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critique")

    graph.add_conditional_edges(
        "critique",
        _critique_router,
        {
            "finalize": "finalize",
            "reformulate": "reformulate",
            "refuse": "refuse",
        },
    )

    # Retry loop: reformulate feeds back into retrieve, not router — the
    # query_type classification from the first pass still holds; only the
    # question text and retrieved chunks need to change on a retry.
    graph.add_edge("reformulate", "retrieve")

    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)

    return graph.compile(checkpointer=checkpointer)


def ask(question: str, thread_id: str, checkpoint_db: str | Path = DEFAULT_CHECKPOINT_DB) -> dict:
    """
    Convenience entry point: run the full graph for a single question on a
    given conversation thread.

    Args:
        question: The user's natural-language question.
        thread_id: Identifies the conversation for checkpointing — state
            (including any future multi-turn context) is scoped to this
            id. Required, not optional, so callers (including the future
            FastAPI layer) are forced to make an explicit choice about
            conversation identity rather than silently sharing one thread.
        checkpoint_db: Path to the SQLite checkpoint file.

    Returns:
        The final graph state dict — use result["final_answer"] for the
        answer text, result["refused"] to check if it was a refusal,
        result["critique_notes"] for the grounding rationale.
    """
    compiled = build_graph(checkpoint_db)
    config = {"configurable": {"thread_id": thread_id}}

    result = compiled.invoke({"question": question}, config=config)
    return result


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Same two test questions as Week 1's llm.py baseline, run through the
    # full graph this time — the second one is the adversarial test that
    # failed in Week 1 and should now be caught by critique_node.
    result1 = ask("How are passwords hashed?", thread_id="manual-test-1")
    print("\n=== Q1: How are passwords hashed? ===")
    print(f"Refused: {result1['refused']}")
    print(result1["final_answer"])

    print("\n" + "=" * 60)

    result2 = ask("What's the rate limit on password reset requests?", thread_id="manual-test-2")
    print("\n=== Q2: What's the rate limit on password reset requests? ===")
    print(f"Refused: {result2['refused']}")
    print(f"Retry count: {result2.get('retry_count')}")
    print(result2["final_answer"])