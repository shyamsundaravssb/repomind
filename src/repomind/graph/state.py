"""
state.py — Shared state schema for the LangGraph pipeline.

One GraphState flows through every node (router -> retrieve -> synthesize
-> critique -> conditional reformulate/end). Fields are additive as the
graph progresses — router sets query_type, retrieve sets chunks, etc. —
rather than each node inventing its own shape, so nodes stay simple
functions of (state) -> partial state update.
"""

from typing import Literal, TypedDict

QueryType = Literal["code_lookup", "how_to", "conceptual"]


class RetrievedChunkDict(TypedDict):
    """
    Plain-dict form of a retrieved chunk, used in GraphState instead of
    the RetrievedChunk dataclass. LangGraph's SqliteSaver checkpoints
    state via msgpack and only trusts a fixed set of built-in/registered
    types — arbitrary dataclasses trigger a deserialization warning now
    and will be a hard error in a future LangGraph version. Plain dicts
    sidestep that entirely and are also directly JSON-serializable, which
    the Week 4 FastAPI layer will want anyway.
    """
    content: str
    metadata: dict
    score: float
    source_collection: str


class GraphState(TypedDict, total=False):
    """
    Full pipeline state. total=False since nodes only need to return the
    keys they actually update — LangGraph merges partial dict returns into
    the running state automatically.
    """

    # --- input ---
    question: str

    # --- router output ---
    query_type: QueryType

    # --- retrieval output ---
    retrieved_chunks: list[RetrievedChunkDict]

    # --- synthesis output ---
    draft_answer: str

    # --- critique output ---
    is_grounded: bool
    critique_notes: str

    # --- retry control ---
    retry_count: int
    max_retries: int

    # --- final output ---
    final_answer: str
    refused: bool