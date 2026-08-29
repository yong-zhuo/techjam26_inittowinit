# Conversational Retrieval Agent — TechJam 2026 Track 4

A multi-turn conversational retrieval system over a frozen 50,000-item Amazon clothing catalog. The agent must surface a hidden target product into a ranked top-10, as high and as early as possible, within 10 turns.

**Current score on the 200 public sessions: `0.829151`** (Hit@10 `0.975`, MRR `0.596835`, MTTC `2.87`), against a provided BM25 baseline of `0.10671`.

---

## Why this scores 8× the baseline

The honest answer, in one sentence: **the baseline never asked the customer anything, so it spent nine of its ten turns receiving no new information.**

The simulated customer is deterministic and discloses on request only. At [`local_evaluator.py:170`](../evaluator/local_evaluator.py#L170), if the agent returns `ask_attribute = None`, the reply is a fixed string — *"Those options are not quite right yet. Ask me about one specific attribute"* — carrying zero information about the target. The starter agent never set that field, so every session after turn 1 was a repeat of turn 1.

Setting `ask_attribute` changes the customer's behaviour: `"other"` ([line 180](../evaluator/local_evaluator.py#L180)) bypasses the type filter and returns the next two undisclosed constraints of any kind. Those constraints are drawn from the target product's own `features` and `details` fields, so they are near-verbatim catalog text. Accumulating them across turns and searching with them directly is what produces the gain.

Three changes account for the whole difference, each measured independently:

1. **Keep a message history.** Retrievers have no memory; searching only the newest message discards everything the customer already said. `0.1067 → 0.2284`.
2. **Ask every turn.** One field, one value. `0.2284 → 0.7504`.
3. **Never re-show an item already offered.** `0.7504 → 0.8292`.

The third deserves its own explanation, below.

### Why "never re-show" is a legitimate behaviour, not a scoring trick

Once the customer has disclosed everything they know, the query stops changing — and so does the answer. Before this change, **every one of the 25 failing sessions burned all 10 turns re-showing the same ten rejected items.** That is a real product defect: a shop that answers "not quite right" by showing you the same shelf again.

The fix is what any shopping interface does when you reject a page of results: show the next page. The agent tracks what it has already offered and returns the highest-ranked items the customer has *not* yet seen. The previously-shown set is cleared on an intent override, because a change of mind invalidates prior rejections.

This is not a scoring exploit. It never reads a label, and it cannot show the target "early" by luck — if the target had been in a previously-shown page, the session would already have ended in a hit. It strictly adds coverage that was previously wasted.

---

## Architecture

```
customer message
  → dialogue state (accumulate constraints, detect and handle override)
  → query construction (full disclosed history)
  → retrieval seam:  retrieve(query, slots, track, top_k)
       └─ BM25 over SQLite FTS5, field-weighted
  → control policy (dedupe, never-empty, unseen-first, always 10 results)
  → 10 ASINs + ask_attribute
```

Retrieval is deliberately behind one frozen function so the sparse implementation can be replaced with a hybrid or reranked one without touching dialogue code.

**Deterministic by design.** Every decision above is hand-written control flow, not model self-planning. It costs zero tokens, runs the full 200-session evaluation in roughly 13 seconds, and can be explained line by line to a judge asking "why did it do that?".

---

## Measured results

### Ablation

Each row changes exactly one thing from the row marked as its base. All runs are the full 200 public sessions with the deterministic evaluator.

| Config | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| BM25 baseline (organizer-provided figure) | 0.125 | 0.068034 | 9.81 | 0.10671 |
| **+ message history** | 0.270 | 0.151381 | 8.60 | 0.228414 |
| **+ ask_attribute = "other"** | 0.875 | 0.540002 | 3.455 | 0.750401 |
| **+ never re-show an offered item** | **0.975** | **0.596835** | **2.87** | **0.829151** |

Rejected variants, each measured against the `0.750401` configuration and **not** shipped:

| Variant | Score | Δ | Why it was cut |
|---|---|---|---|
| Query reordered, newest-first | 0.750401 | 0.000 | BM25 ignores word order; the 40-term cap it was meant to dodge is hit by only 1.8% of queries |
| Drop no-information customer replies | 0.736251 | −0.014 | Removing terms narrowed BM25's match set more than it removed noise |
| Erase superseded history on override | 0.736494 | −0.014 | See below — the "abandoned" preference is still true of the target |
| Both of the above | 0.715657 | −0.035 | |
| Profile seeding (`preference_tags` into query) | 0.737643 | −0.013 | Tags are generic (`fit`, `comfort`, `durability`) and dilute the query |

Profile seeding is retained behind `PROFILE_SEED=1`, off by default, exactly as the design predicted it might need to be.

**The override finding is worth stating plainly**, because it contradicts the standard dialogue-state assumption. In an intent-override session the customer says *"actually, ignore my earlier preference"* — but the evaluator constructs that earlier preference from the target product's own `soft_preferences` ([`local_evaluator.py:79`](../evaluator/local_evaluator.py#L79)). The abandoned constraint is still a true description of the product being hunted. Forgetting it destroys evidence. The agent therefore detects overrides (30/30 genuine overrides caught) and uses them to reset the shown-item set, but does **not** erase the accumulated query.

### Per scenario, final system

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.9625 | 0.541161 | 2.475 |
| browsing | 80 | 0.9875 | 0.582510 | 2.6125 |
| intent_override | 30 | 0.9667 | 0.705410 | 4.3667 |
| boundary | 10 | 1.0000 | 0.831111 | 3.600 |

Browsing was the largest hole in the baseline (Hit@10 `0.025`, 40% of all sessions) and is now the strongest category. Browsing sessions open with *"I'm looking for X, but I'm still exploring"* and disclose nothing — so they were entirely starved by an agent that never asked.

### Failure analysis

At the `0.750401` configuration, all 25 misses were diagnosed by re-running each session's final query to depth 2000:

| Cause | Count |
|---|---|
| Target retrieved but ranked below 10 | **25** |
| Target never retrieved | **0** |

Not one failure was a retrieval failure. Recall by depth, over the same runs:

| Depth | Ceiling on Hit@10 |
|---|---|
| top 10 | 0.875 |
| top 30 | 0.925 |
| top 100 | 0.975 |
| top 500 | **1.000** |

**BM25 alone already finds every target within its top 500.** Every remaining point is an ordering problem, not a recall problem. The five sessions still missed at `0.829151` are those whose targets sit at ranks 110, 110, 125, 146 and 315 — beyond the 100 items reachable in 10 turns of 10.

This measurement sets the priority for the rest of the project: reranking, not recall. It also fixes the depth a reranker must consider — reranking only the top 30 would cap Hit@10 at 0.925, below where the system already is.

---

## Reproducing

```bash
pip install -r requirements.txt
cd ../.. && python -m evaluator.local_evaluator
```

The catalog is not committed. Download `catalog.jsonl.gz` from the challenge GitHub Release and decompress it to `data/catalog.jsonl` (50,000 rows). Its location is read from `CATALOG_PATH`, defaulting to `data/catalog.jsonl`.

`python run_eval.py --note "what changed"` runs the same evaluation and appends the metrics, commit hash and note to `runs.log`, which is how every row in the ablation table above was produced.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CATALOG_PATH` | `data/catalog.jsonl` | Catalog location |
| `ASK_POLICY` | `other` | `none` disables clarification, reproducing the ablation row |
| `EXPLORE` | `1` | `0` disables unseen-first selection |
| `PROFILE_SEED` | `0` | `1` seeds the query with profile preference tags |
| `DEPTH` | `120` | Candidates requested per turn |

`DEPTH` is not a tuned value. At most 10 items are shown per turn across at most 10 turns, so any value ≥ 100 is equivalent; the score is identical at 100, 120, 200, 500 and 1000.

### Network and cost

The system as measured makes **no network calls at runtime and uses zero LLM tokens**. It reports `prompt_tokens: 0, completion_tokens: 0`, and a full 200-session evaluation completes in about 13 seconds on a laptop, including building the index.

---

## Limitations

- **Ordering is the bottleneck.** Hit@10 is capped at 0.975 by what is reachable in 100 slots; closing the rest requires better ranking, not better search.
- **Slot state is currently inert.** `SlotState` accumulates colour and material and detects overrides, but the query is built from raw message history, so slot values do not yet influence the score. It is wired for a reranker that consumes a structured constraint dict.
- **Single-valued slots misrepresent real products.** Five of the 35 override detections were false positives caused by products listing several materials (*"Polyester,Cotton,Spandex"*), which a one-value `material` slot reads as self-contradiction. Harmless today; it must be fixed before slots feed ranking.
- **Attribute extraction covers colour and material only**, by regex over a closed vocabulary. The catalog has no `color` or `material` field — both exist only in free text.
- **Tuned and measured on the 200 public sessions.** The private 800 share the same generator and scenario mix, but no held-out confirmation is possible before submission.

---

## To be completed

Sections owned by Person A, to be merged before submission: model choice and rationale, `requirements.txt` with pinned versions, `setup.sh`, dense-retrieval and fusion results, reranking cost disclosure (latency mean/p95, tokens, estimated spend), and behaviour without API credentials.

## Team contributions

- **Person A** — retrieval: BM25 index, dense retrieval, fusion, reranking, index build.
- **Person B** — dialogue: control policy, state tracking, query construction, clarification policy, guardrails, evaluation and failure analysis.
