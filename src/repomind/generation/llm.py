"""
llm.py — Groq/Llama client and basic RAG answer chain (Week 1 baseline).

Wraps a single Groq chat completion call with a prompt that instructs the
model to answer only from the provided context and explicitly refuse when
the context doesn't contain the answer — this is the Week 1 stand-in for
the Week 2 critique/retry node, not a replacement for it. There's no
self-critique or reformulation loop here; that's graph-level logic that
comes later.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a codebase Q&A assistant. Answer the user's question using ONLY the context provided below — do not use any outside knowledge about the library or framework in question.

Rules:
- If the context fully answers the question, answer clearly and cite which file/function or doc section you used.
- If the context is documentation describing something (e.g. a function) that does NOT actually appear in the code context provided, say so explicitly — do not assume it exists just because it's documented.
- If the context does not contain enough information to answer the question, say clearly that the codebase/docs provided don't contain this information. Do not guess or fabricate function names, parameters, or behavior.
"""


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Construct a ChatGroq client.

    Args:
        temperature: Sampling temperature. Defaults to 0.0 for
            deterministic, grounded answers — appropriate for a Q&A
            system where hallucination is the primary failure mode being
            guarded against, not creativity.

    Returns:
        A configured ChatGroq instance.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment/.env.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file (see .env.example)."
        )

    return ChatGroq(model=MODEL_NAME, temperature=temperature )



def answer_question(question: str, context: str) -> str:
    """
    Answer a question given pre-retrieved context, using the grounding-
    focused system prompt above.

    Args:
        question: The user's natural-language question.
        context: Formatted context string (see retriever.format_context).

    Returns:
        The model's answer as plain text.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    from repomind.retrieval.retriever import format_context, retrieve

    question = "How are passwords hashed?"
    chunks = retrieve(question, k=5)
    context = format_context(chunks)

    print(f"Question: {question}\n")
    print(answer_question(question, context))

    print("\n" + "=" * 60)

    # Adversarial test: this should trigger refusal / explicit
    # doc-vs-code gap flagging, not a fabricated answer.
    question2 = "What's the rate limit on password reset requests?"
    chunks2 = retrieve(question2, k=5)
    context2 = format_context(chunks2)

    print(f"\nQuestion: {question2}\n")
    print(answer_question(question2, context2))