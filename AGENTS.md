# AGENTS.md

TikTok TechJam 2026 Track 4 — a conversational retrieval agent over a frozen 50,000-item Amazon catalog. Two people, ~50 working hours.

Part of a set: `AGENTS.md` (always-loaded rules and commands), `PLAN.md` (roadmap and gates), `DESIGN.md` (component rationale and research), `SUBMISSION.md` (packaging and rules).

Read this file on every task. Read `PLAN.md` before starting a phase, `DESIGN.md` when building a named component, `SUBMISSION.md` before Phase 5.

---

## Frozen paths — never edit

| Path | Why |
|---|---|
| `evaluator/` | Official scorer. Modifying it invalidates the run |
| `data/` | Frozen catalog and public sessions. Read-only, no mutations |
| `docs/` | Contract and config. Authoritative reference |

`docs/agent_api_contract.json` is authoritative for response shape and the `ask_attribute` enum. Verify against it, not memory.

## Never do

Fine-tune or train any model. Deploy an external vector database. Add Redis or any external datastore. Add LangChain, LlamaIndex, or any agent framework. Build a web UI. Exceed 10 turns in a session.

**Never commit:** the catalog, any `*.jsonl`, `.env`, API keys, or model weights.

Before every commit:
```bash
git log -p | grep -iE "api_key|sk-|AIza|gsk_"
```

## Layout

Build in `starter/`. The evaluator imports `from starter.agent import Agent`, so this needs no shim.

```
starter/
├── agent.py                exports class Agent — control policy
└── src/
    ├── index_build.py      offline: writes embeddings.npy + asins.json
    ├── dialog/
    │   ├── state.py        SlotState — accumulate, override, erase
    │   └── query.py        query rewriting and buying/browsing routing
    ├── retrieval/
    │   ├── interface.py    retrieve() — the frozen seam
    │   ├── sparse.py       BM25 over SQLite FTS5
    │   ├── dense.py        bge-small bi-encoder, brute-force cosine
    │   ├── fusion.py       reciprocal rank fusion
    │   └── rerank.py       listwise LLM rerank, falls back to fused order
    └── obs/cache.py        disk cache keyed on prompt hash
```

There are no agent tests yet. `tests/` at the repo root holds the kit's own evaluator tests.

Full target layout, including files not yet built, is in `SUBMISSION.md`.

## The frozen interface

```python
# starter/src/retrieval/interface.py
def retrieve(query: str, slots: SlotState, track: str, top_k: int) -> list[str]:
    """Ranked parent_asins, best first. Never returns an empty list."""
```

Frozen after Phase 0. `SlotState` carries `color` and `material`; adding or removing a field, like changing `retrieve()`'s signature, is a conversation, not a commit.

## Ownership

| | Person A | Person B |
|---|---|---|
| Owns | `src/retrieval/`, `index_build.py`, `src/obs/cache.py` | `agent.py`, `src/dialog/` |
| Metric | Hit@10, recall@500 | MTTC, zero crashed sessions |

Person B does not edit `src/retrieval/`. Person A does not edit `agent.py`. Separate branches, merge at phase checkpoints. Both run the evaluator independently.

## Commands

```bash
python -m evaluator.local_evaluator          # writes results.json
python run_eval.py --note "what changed"     # above, plus a row in runs.log
python -m pytest tests -q                    # the kit's evaluator tests
```

There is no `make` on the Windows dev machine; `run_eval.py` replaces `make eval`.

## Rules of work

- **Change one thing between evaluator runs.** The evaluator is deterministic; that is worthless if score movements cannot be attributed.
- **Log every run** with `run_eval.py --note`, so each number carries its commit.
- **Never tune the reranker while recall is broken.** See the debugging ladder in `PLAN.md`.
- **Nothing in `starter/src/` may import from outside `starter/`**, and no `../` paths.
- Catalog path comes from `CATALOG = os.getenv("CATALOG_PATH", "data/catalog.jsonl")`, never hardcoded.
- Comments: none by default, at most one short `#` line. No block comments, no multi-line docstrings, no phase numbers in source.

## Measured facts

Verified by inspection, not assumed:

- **The customer only reveals information when asked.** `local_evaluator.py:170` — with `ask_attribute = None` the simulator discloses nothing for the whole session. `"other"` (line 180) returns the next two undisclosed constraints of any type. Measured over 200 sessions with BM25 only: keeping a message history and setting `ask_attribute` moves the score from `0.1067` to `0.7504`. Build those before any retrieval work. Full table in `PLAN.md`.
- **Baseline:** Hit@10 `0.125`, MRR `0.068034`, MTTC `9.81`, composite `0.10671`. Reproduced exactly.
- **Per scenario:** buying `0.2375` (80 sessions), intent_override `0.1333` (30), browsing `0.025` (80), boundary `0.0` (10). Browsing is 40% of sessions and the largest hole.
- **Sessions carry `scenario_type` labels**, and the evaluator reports per-scenario metrics for free. The scenario table is possible.
- **Catalog fields:** `parent_asin, title, features, description, price, categories, details, average_rating, rating_number, store`. There is **no `color` or `material` field** — both live in free text only, so attribute extraction is required. `price` can be `null`.
- **The FTS5 BM25 index builds in ~1.4s** over 50k products. Fast enough; no need to swap to `bm25s`.
- **The evaluator silently swallows agent exceptions** (`local_evaluator.py:241`) and substitutes an empty recommendation list. A crash is an invisible miss, never an error message.
- **Slot extraction covers ~43% of disclosable constraints.** Much of the remainder is boilerplate (`Imported`, `Machine Wash`) that should not become slots.

## Flag drift, do not silently comply

Say something when work departs from `PLAN.md`. Name the drift, name the cost, propose the alternative, then proceed as directed. A confirmed override is a decision, not drift.

Flag when: a phase gate is unmet and work continues anyway (especially Hit@10 below 0.125 with the next task not being recall); a frozen path is being edited; an excluded technology appears; something requires runtime network with no fallback; more than one thing changed between evaluator runs; the frozen interface is being modified; ownership is crossed; presentation time drops below eight hours; a component is built without an eval run proving it moved a number.
