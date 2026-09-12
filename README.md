# RepoMind

A multi-agent, LangGraph-orchestrated Q&A system for codebases, built to demonstrate production-quality RAG engineering: structure-aware retrieval, self-critique/retry grounding, and a rigorous evaluation methodology (LangSmith + RAGAS) with quantitative ablation studies.

The differentiator isn't the chatbot — it's the evaluation discipline. Every design decision below is backed by a measured before/after number, not just "it works."


---

## What it does

RepoMind answers four categories of questions about a codebase:
- **Factual code lookup** — "What hashing algorithm does `hash_password` use?"
- **Procedural / how-to** — "How do I safely release a database connection back to the pool?"
- **Conceptual / architectural** — "How does the login lockout policy work end-to-end?"
- **Adversarial** — questions where the documentation describes something that isn't actually implemented in code. Correct behavior is refusal or an explicit doc/code mismatch flag, not confident fabrication.

The system routes each query, retrieves from separate code and docs indexes, drafts an answer, critiques its own groundedness against the retrieved code (not just the docs), and retries with a reformulated query before giving up and refusing.

---

## Architecture

**Ingestion (offline)**
- Code parsed with Python's `ast` module — one chunk per top-level function/class, so a chunk is always a complete semantic unit rather than an arbitrary character window.
- Markdown docs chunked by `#`/`##` headers.
- Two separate Chroma collections (`code_chunks`, `doc_chunks`) rather than one merged index — this is what makes the router/multi-retriever architecture possible, and what makes the ablations below a clean single-variable comparison.
- Embeddings: `BAAI/bge-small-en-v1.5`, CPU-only, with the required manual query-instruction prefix for correct asymmetric search.

**Runtime graph (LangGraph)**

```
router → retrieve → synthesize → critique
                                    │
                    ┌───────────────┼───────────────┐
                 grounded      ungrounded,        ungrounded,
                    │           retries left      retries exhausted
                    ▼               │                   │
                finalize      reformulate            refuse
                    │               │                   │
                   END          (loop to retrieve)      END
```

- **Router** — classifies the query into `code_lookup` / `how_to` / `conceptual`, which weights how many chunks are pulled from each collection.
- **Retrieve** — parallel, weighted retrieval from both collections (this is the "router + multi-retriever" architecture — see [Retrieval architecture ablation](#3-retrieval-architecture-ablation-not-yet-run)).
- **Synthesize** — drafts an answer from retrieved context only, instructed to flag (not assume) any feature described only in documentation.
- **Critique** — the core grounding mechanism. An LLM judges whether the draft's claims are corroborated by the *code* portion of the context, not just the docs — plus a cheap programmatic backstop: if zero code chunks were retrieved at all, the answer is forced ungrounded regardless of the LLM's verdict. This two-layer check was added after the initial LLM-only critique passed an answer that was accurately quoting documentation for a function that doesn't exist in the code (see [Adversarial refusal test](#adversarial-refusal-test)).
- **Reformulate / retry** — on ungrounded verdicts, the question is rewritten and retried (max 2 retries) before falling through to an explicit refusal.

**Checkpointing:** LangGraph `SqliteSaver`, so multi-turn state survives process restarts.

---

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Framework | LangChain |
| Vector store | ChromaDB (two collections: code, docs) |
| Embeddings | `BAAI/bge-small-en-v1.5` (CPU-only) |
| LLM | Groq, `openai/gpt-oss-120b`¹ |
| Observability & eval | LangSmith (tracing, dataset versioning, experiment comparison) |
| Eval metrics | RAGAS (faithfulness, context precision, context recall) + custom deterministic/LLM-judge evaluators |
| Persistence | LangGraph `SqliteSaver` |
| Backend | FastAPI *(in progress)* |

¹ Originally spec'd as Llama 3.3 70B; Groq deprecated that model mid-project. Swapped to `openai/gpt-oss-120b`, Groq's recommended free-tier replacement. See `DECISIONS.md` for the full record — this is a generation-only change and does not affect ingestion, chunking, retrieval, or eval methodology, though RAGAS numbers below are specific to this model.

---

## Adversarial refusal test

The sample repo has a deliberate gap: `docs/auth.md` documents `request_password_reset()` and `complete_password_reset()` in detail (including a specific "3 requests per hour" rate limit), but neither function exists in `auth.py`. This is the project's built-in test for hallucination under documentation pressure.

| Stage | Behavior on "What's the rate limit on password reset requests?" |
|---|---|
| Week 1 (prompt-only grounding instruction) | **Failed.** Answered confidently from docs alone ("3 requests per hour"), citing only `auth.md`, no flag that no corresponding code exists. |
| Week 2 (critique node, LLM-only) | **Failed initially.** Critique judged the answer "grounded" because it accurately quoted the documentation — accurate quoting isn't the same as code-side corroboration. |
| Week 2 (critique node, hardened) | **Passes.** Prompt rewritten to explicitly require code-side corroboration for any feature claim; programmatic backstop added (forces ungrounded if zero code chunks were retrieved). Correctly reformulates twice, then refuses, citing that no corresponding function exists in the retrieved code. |

This before/after is the concrete justification for the two-layer critique design, not just a design preference.

---

## Results

### Week 3 — Baseline evaluation (AST chunking, full graph)

21-question eval set (`repomind-eval-v1`): 6 code-lookup, 6 how-to, 6 conceptual, 3 adversarial.

**Correctness pass — 21/21 complete:**

| Metric | Result |
|---|---|
| `grounding_correctness` / `refusal_correctness` (non-adversarial, 18 questions) | 18/18 (100%) |
| `no_hallucination` (adversarial, 3 questions) | 3/3 (100%) |

Note: `refusal_correctness` alone understates system quality on adversarial questions — it only credits the *hard-refuse* path, but a critique that honestly states "this isn't in the code" on the first pass (without needing to exhaust retries) is equally correct and non-hallucinating. `no_hallucination` was added specifically to measure the thing that actually matters (did it fabricate a fact), independent of which graph path got there. Which of the 3 adversarial questions hits hard-refusal vs. an honest first-pass answer varies run to run — this is expected non-determinism in the critique LLM's judgment, not a regression.

**RAGAS pass — 15/21 (71%) coverage, accepted as final baseline:**

| Metric | Value (15/21 valid rows) |
|---|---|
| `ragas_faithfulness` | ~0.75 (range 0.28–0.98) |
| `ragas_context_precision` | ~0.87 |
| `ragas_context_recall` | ~0.93 |

Coverage is capped by Groq's free-tier daily token limit (200,000 TPD on `openai/gpt-oss-120b`) — a full RAGAS pass costs roughly 12,000–13,000 tokens/question across all three metrics, which structurally exceeds the daily cap for 21 questions regardless of when the run starts (confirmed across 5 attempts on separate days, including a verified fresh-quota run). This is a quota ceiling, not a retry/ordering bug. Context recall is skewed upward by non-adversarial questions scoring 1.0; the two adversarial questions that did score show notably lower recall (0.33–0.5), consistent with there being little to retrieve for a genuinely unanswerable question.

---

### Week 4 — Ablation 1: AST-aware vs. naive fixed-size chunking

**Design:** retrieval architecture (Week 2 router + critique + retry graph) held constant; only code-chunking strategy varies. Naive chunker uses `RecursiveCharacterTextSplitter` sized to the AST chunker's real median output (445 chars) so chunk *size* isn't a confound — only boundary-awareness is. Docs chunking is identical in both arms.

**Correctness pass — 21/21 complete:**

| Metric | AST chunking (baseline) | Naive chunking |
|---|---|---|
| Non-adversarial correctness (18 questions) | **18/18 (100%)** | **15/18 (83%)** |
| Adversarial `no_hallucination` (3 questions) | 3/3 (100%) | 3/3 (100%) |
| Overall `grounding_correctness` | 18/21 | 16/21 |

**Finding:** naive chunking produced 3 false refusals on genuinely answerable non-adversarial questions — `decode_access_token`'s return semantics, `create_access_token`'s signing logic, and the `retry()` decorator's implementation. In each case the fixed 445-character window cut a function off mid-body (after its docstring but before its implementation), so the critique node's code-corroboration check correctly found no complete evidence in the retrieved chunk and refused — even though the real answer existed in the file, just split across a chunk boundary that wasn't retrieved alongside it.

Hallucination-avoidance held at 3/3 in both arms — naive chunking didn't make the system more likely to fabricate an answer, it made it more likely to *unnecessarily refuse a correct one*. This is a precision/recall-style trade-off, not a safety regression: **AST-aware chunking reduces false refusals without costing hallucination safety.**

**RAGAS pass — in progress, blocked on Groq daily quota** (same structural ceiling as the baseline; expect similar partial coverage, resumable via evaluator-only reruns rather than full graph re-invocation).

---

### Ablations not yet run

| Ablation | Status |
|---|---|
| No reranker vs. cross-encoder reranker (`bge-reranker`) | Not started — no reranker code written yet |
| Single merged retriever vs. router + multi-retriever architecture | Not started — both arms already exist (Week 1 baseline retriever vs. Week 2 graph); just needs eval runs with distinct experiment prefixes |

---

## Known limitations

- `issue_history` as a fourth router category is deferred — no issues/PR retriever has been built, so routing into that category would be a dead end.
- Code chunking is Python-only (stdlib `ast`); multi-language support would need tree-sitter.
- One RAGAS eval anomaly is unresolved: example `6051c007` (a timestamp-formatting question) consistently returns `status=success` with blank scores on all three RAGAS metrics, distinct from the rate-limit failures. Not yet investigated; likely an empty `retrieved_contexts` or reference-field edge case.
- RAGAS coverage on both the baseline and the chunking ablation is capped at roughly 15/21 (71%) per run by Groq's free-tier daily token limit, not by retry logic — see the Results sections above for what that means for the reported numbers.

---

## Project structure

```
repomind/
├── sample_repo/                  # hand-written toy codebase (auth.py, database.py, utils.py + docs)
│                                  #   deliberate doc/code gap = adversarial refusal test case
├── src/repomind/
│   ├── ingestion/                # AST + naive code chunkers, doc chunker, embeddings, vectorstore
│   ├── retrieval/                # Week 1 baseline merged retriever (permanent ablation comparison point)
│   ├── graph/                    # LangGraph nodes, state, build/compile, ask() entry point
│   ├── generation/                # LLM client
│   ├── eval/                     # eval dataset, LangSmith evaluators, run_eval.py
│   └── api/                      # FastAPI backend (in progress)
├── data/                         # Chroma stores (gitignored): chroma_db/, chroma_db_naive_chunking/
├── scripts/                      # run_ingestion.py, run_ingestion_naive.py
└── DECISIONS.md                  # full design-decision log with rationale and trade-offs
```

---

## Development approach

Built and validated end-to-end against a hand-written `sample_repo/` before pointing at a real target repository (FastAPI), to decouple pipeline-correctness debugging from large-codebase complexity. Swapping in the real repo is designed to be a path-config change only. Every ablation isolates a single variable — architecture, chunking, or reranking are never changed simultaneously — so the results above are directly attributable to the one thing that changed. Full rationale for every design decision, including forced substitutions (model deprecation, dependency workarounds, rate-limit handling) and trade-offs explicitly *not* taken, is in `DECISIONS.md`.
