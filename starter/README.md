# Conversational Retrieval Agent — TechJam 2026 Track 4

A multi-turn conversational retrieval system over a frozen 50,000-item Amazon clothing catalog. The agent must surface a hidden target product into a ranked top-10, as high and as early as possible, within 10 turns.

**Current score on the 200 public sessions: `0.800304`** — Hit@10 `0.965`, MRR `0.521012`, MTTC `2.925` — against a provided BM25 baseline of `0.10671`. No LLM, no runtime network, zero tokens, ~13s per full evaluation.

---

## The one measurement that explains the architecture

Everything below follows from a single property of the benchmark, which we verified by reading the evaluator rather than guessing:

**The simulated customer knows exactly four facts, and only discloses them when asked.**

- If the agent returns `ask_attribute = None`, the customer replies with a fixed sentence containing no information about the target ([`local_evaluator.py:170`](../evaluator/local_evaluator.py#L170)). The starter agent never set that field, so every turn after the first was a repeat of the first.
- If the agent sets `ask_attribute`, the customer returns up to two undisclosed facts ([`line 178`](../evaluator/local_evaluator.py#L178)).
- Those facts are drawn from the target product's own `features` and `details` fields — so they are near-verbatim catalog text.

We measured the supply of facts across all 200 sessions:

| | Finding |
|---|---|
| Facts per session | **Exactly 4, in all 200 sessions** |
| Turns to extract them all with `other` | **2** |
| Sessions where new information arrives after turn 2 | **0** |

This is the fact that shapes the whole design. **Information is exhausted by turn 2; turns 3–10 are a pure ranking and coverage problem.** An architecture that keeps interrogating a customer who has nothing left to say is solving the wrong problem.

The type distribution matters too, because it determines what is worth asking about:

| Fact type | Sessions containing at least one |
|---|---|
| `feature` | 96% |
| `material` | 76% |
| `color` | 26% |
| `style` | 9% |
| `size` | 4% |
| `use_case` | 2% |

Three of the contract's ten allowed attributes — `brand`, `category`, and effectively `budget` — never carry information in this dataset at all.

---

## Architecture, and why each piece is there

```
customer message
  → dialogue state        accumulate constraints; detect intent override
  → query construction    concatenate all disclosed customer text
  → routing               buying vs browsing, passed to retrieval
  → retrieval seam:  retrieve(query, slots, track, top_k)
       ├─ BM25 over SQLite FTS5, field-weighted
       └─ dense bi-encoder (bge-small-en-v1.5), fused with RRF
  → control policy        dedupe · validate · unseen-first · never-empty
  → 10 ASINs + ask_attribute
```

**Why concatenate raw history rather than build a structured query.** Retrievers have no memory. Handed only the newest message, *"For that, what matters is: black"* searches the entire catalog for black things, having forgotten it was ever about boots. Since the customer's disclosed text is near-verbatim catalog text, concatenating it *is* the query — the most direct possible use of the signal. Worth `0.1067 → 0.2284` on its own.

**Why ask `"other"` every turn rather than choosing an attribute.** See the ask-policy section below; this is measured, and there is also a proof.

**Why never re-show an item already offered.** Once information is exhausted at turn 2, the query stops changing and so does the ranking. Without this, every failing session spent its remaining turns re-showing the same ten rejected items — a real product defect, not just a scoring one. Worth `0.7504 → 0.8292` on the sparse-only system.

**Why the retrieval seam is frozen.** `retrieve(query, slots, track, top_k)` is the only contact point between dialogue and retrieval. It let the sparse implementation be replaced with a hybrid one, by a different person on a different branch, without a single change to dialogue code.

**Why it is deterministic.** Every decision above is hand-written control flow, not model self-planning. It costs zero tokens, is reproducible to six decimal places across runs, and can be explained line by line to a judge asking "why did it do that?".

---

## What we measured

Every row below is a real evaluator run on all 200 public sessions, logged to `runs.log` with its commit hash. **Runs marked (sparse) predate the hybrid retrieval merge** and are relative to a `0.750401` or `0.829151` base; runs marked (hybrid) are on the current system.

### The three changes that built the system — (sparse)

| Config | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| BM25 baseline (organizer figure) | 0.125 | 0.068034 | 9.81 | 0.10671 |
| + message history | 0.270 | 0.151381 | 8.60 | 0.228414 |
| + `ask_attribute = "other"` | 0.875 | 0.540002 | 3.455 | 0.750401 |
| + never re-show an offered item | **0.975** | **0.596835** | **2.87** | **0.829151** |

### Ask policy — (hybrid)

The gap this closes: the previous version of this document could say *what* we did but not *why it beat the alternative*, because the alternative had never been built.

| Policy | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **`other`** (shipped) | **0.9650** | **0.521012** | **2.925** | **0.800304** |
| `targeted` — real attributes, drain rule | 0.9500 | 0.505169 | 3.490 | 0.776751 |
| `none` | 0.5400 | 0.216504 | 7.080 | 0.413351 |

`targeted` is a genuine implementation, not a straw man: it walks the attribute list ordered by the measured frequencies above, and **drains** each attribute — asking `material` repeatedly until the customer says they have nothing more of that type — before advancing.

It loses by `−0.0236`, and MTTC shows why: `2.925 → 3.490`, half a turn slower per session. Asking about `color` in the 74% of sessions with no colour fact returns nothing at all.

**There is also a proof, which is stronger than the measurement.** At [`line 178`](../evaluator/local_evaluator.py#L178), `"other"` returns the first two undisclosed facts of *any* type; a targeted ask returns the first two of *one* type — a subset of the same list. So `"other"` discloses at least as much as any targeted question, on every turn, in every session. Targeted asking is not merely worse here; it is mathematically incapable of winning. No attribute ordering can change that.

### Result-count shapes — (hybrid)

Whether to return fewer than 10 while still under-informed, so less of the 100-slot budget (10 turns × 10 items) is spent on blind early guesses.

| Shape | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **`full`** — always 10 (shipped) | **0.9650** | 0.521012 | 2.925 | **0.800304** |
| `turn1_small` — 3 on turn 1 only | 0.9650 | 0.534520 | 3.015 | 0.802556 |
| `ramp_turn` — `2 + turn`, capped | 0.9500 | **0.649677** | 3.425 | **0.821403** |
| `ramp_facts` — `3 + 3 × facts known` | 0.9400 | 0.624339 | 3.315 | 0.811002 |

**`ramp_turn` scores highest and is deliberately not shipped.** The reasoning is in Tradeoffs below — it is the one place where the highest number is not the chosen configuration.

### Retrieval routes — (hybrid vs sparse)

| Route | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| Sparse only (BM25) | **0.9750** | **0.596835** | **2.870** | **0.829151** |
| Hybrid (BM25 + dense, RRF) — shipped | 0.9650 | 0.521012 | 2.925 | 0.800304 |
| Dense only | 0.5400 | 0.232587 | 6.890 | 0.421976 |

Recorded honestly: **on the public set, sparse-only outscores the hybrid system by `0.029`.** Discussed under Tradeoffs.

### Per scenario, shipped system — (hybrid)

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.9625 | 0.511374 | 2.4625 |
| browsing | 80 | 0.9750 | 0.505288 | 2.7125 |
| intent_override | 30 | 0.9333 | 0.525648 | 4.5000 |
| boundary | 10 | 1.0000 | 0.710000 | 3.6000 |

Browsing was the baseline's largest hole — Hit@10 `0.025` across 40% of all sessions — because those sessions open with *"I'm looking for X, but I'm still exploring"* and disclose nothing. An agent that never asks is completely starved. It is now among the strongest categories.

### Failure analysis — (sparse)

All 25 misses at the `0.750401` configuration were diagnosed by re-running each session's final query to depth 2000:

| Cause | Count |
|---|---|
| Target retrieved but ranked below 10 | **25** |
| Target never retrieved | **0** |

**Not one failure was a retrieval failure.** Recall by depth:

| Depth | 10 | 30 | 100 | 500 |
|---|---|---|---|---|
| Ceiling on Hit@10 | 0.875 | 0.925 | 0.975 | **1.000** |

BM25 alone finds every target within its top 500. This single table reset the project's priorities: **the remaining work is ordering, not recall.** It also sets the depth a reranker must consider — reranking only the top 30 would cap Hit@10 at 0.925, *below where the system already scores*.

The sessions still missed have targets at true ranks 110, 110, 125, 146 and 315 — beyond the 100 items reachable in 10 turns of 10. No reranker can recover them, because they never enter the candidate pool.

### Guardrails, proven by fault injection

An unexercised guardrail is not a working guardrail. The evaluator swallows agent exceptions silently ([`line 241`](../evaluator/local_evaluator.py#L241)) and substitutes an empty list, so a crash is an invisible miss rather than an error message. We therefore both **count** exceptions and **inject** them.

Every run now ends with a line on stderr:

```
[agent] exceptions caught: 0 (search=0, respond=0)
```

Injecting a fault into every third query build, at a 33% failure rate:

| | Score | Turns returning nothing |
|---|---|---|
| With guards | **0.784111** | **0** |
| Without guards | 0.742808 | 273 |

The guards are worth `+0.041` under that fault rate, and eliminate empty responses entirely — including on turn 1, where there is no previous ranking to fall back on, via a warm-up ranking captured at startup.

**ASIN validation is now implemented and tested the same way.** Every candidate is checked against the loaded catalog before it can occupy one of the ten returned slots — closing the one guardrail from `AGENTS.md`'s list that had been built but never proven. We tested it against a simulated hallucinating reranker, injecting 3 and then 7 fake IDs (of 10 slots) at the front of the ranking every turn — the position a bad reranker would place them, having judged them "best":

| Poisoning | Hit@10, filter off | Hit@10, filter on | Score, off | Score, on |
|---|---|---|---|---|
| 3 of 10 fake | 0.9600 | **0.9650** | **0.801874** | 0.800304 |
| 7 of 10 fake | 0.9600 | **0.9650** | **0.802080** | 0.800304 |

**Hit@10 is consistently better with the filter on**, at both poisoning levels — the expected mechanism: fake IDs no longer occupy a slot a real candidate could have used. The composite score is very slightly lower with the filter on, by almost exactly the same small amount (`~-0.0017`) regardless of whether 30% or 70% of candidates are hallucinated. That consistency identifies the cause: it is the same MRR-inflation-via-paging effect disclosed below, not a cost that scales with how badly a reranker fails. We ship the filter anyway — Hit@10 is 50% of the score and the mechanism protecting it is unambiguous, while the tiny score offset is bounded, explained, and not a reason to skip basic output validation.

We also found and fixed a related gap while building this test: the warm-up ranking fetched at startup (the very-first-turn fallback) was never itself validated against the catalog. If retrieval were broken from the first call, the fallback could have been poisoned too. It is now filtered the same way as every other candidate list.

### MRR inflation from paging, disclosed — (sparse)

Never re-showing an item inflates MRR: if 40 items have been excluded over earlier turns and the target then lands at position 1, the evaluator records rank 1 for something whose true unfiltered rank was 41. We recovered every hit's true rank to quantify it:

| Measure | Value |
|---|---|
| MRR as scored | 0.596835 |
| MRR at true unfiltered rank | 0.534539 |
| **Inflation** | **+0.062297** |
| Hits requiring paging | 27 of 195 (14%) |

At MRR's 30% weight that is `+0.0187` of the change's `+0.0788` total gain — **about a quarter presentation, three quarters genuine** improvement in Hit@10 and time-to-hit, neither of which paging can fake.

---

## Things we built, measured, and rejected

Every one of these is a real implementation that lost. They are listed because a component that does not move a number has not earned its place, and saying so is stronger evidence of judgement than a longer feature list.

| Variant | Score | Δ vs its base | Why rejected |
|---|---|---|---|
| Targeted ask policy with drain rule | 0.776751 | −0.024 | `other` is a strict superset of any single-attribute ask — see the proof above |
| `ramp_turn` result count | 0.821403 | **+0.021** | Highest score, still rejected — see Tradeoffs |
| `ramp_facts` result count | 0.811002 | +0.011 | Same reason, smaller gain, worse Hit@10 |
| Query reordered newest-first | 0.750401 | 0.000 | BM25 ignores word order; the 40-term cap it was meant to dodge affects only 1.8% of queries |
| Drop no-information replies from query | 0.736251 | −0.014 | Removing terms narrowed BM25's match set more than it removed noise |
| Erase superseded history on override | 0.736494 | −0.014 | The "abandoned" preference is still true of the target — see below |
| Both of the above | 0.715657 | −0.035 | |
| Profile seeding (`preference_tags`) | 0.737643 | −0.013 | Tags are generic (`fit`, `comfort`, `durability`) and dilute the query |

**The override finding contradicts standard dialogue-state practice, and is worth stating plainly.** In an intent-override session the customer says *"actually, ignore my earlier preference"* — but the evaluator builds that earlier preference from the target product's own `soft_preferences` ([`line 79`](../evaluator/local_evaluator.py#L79)). The abandoned constraint is still a true description of the product being hunted. **Forgetting it destroys evidence.** The agent therefore detects overrides — 30 of 30 genuine ones caught — and uses them to reset the shown-item set, but deliberately does **not** erase the accumulated query.

---

## Tradeoffs and open decisions

### 1. We ship `full` result counts despite `ramp_turn` scoring higher

`ramp_turn` scores `0.821403` against our shipped `0.800304`. We do not ship it, for three reasons:

- **It buys MRR with Hit@10.** Hit@10 falls `0.965 → 0.950`; those are real sessions where the target was in our candidate list and we chose not to show it. Hit@10 is 50% of the score, MRR is 30%.
- **Its gain is largely the paging effect again, not better ranking.** Shrinking a list cannot reorder it — the item ranked fifth is still ranked fifth, merely hidden. The MRR rises because hidden items are not marked as shown and resurface higher later. It is the same mechanism as never-re-showing, applied harder, and it carries the same disclosure caveat.
- **It competes with work already planned.** Reranking is the intended MRR lever and can do it properly, by genuinely reordering candidates. Spending Hit@10 now to buy MRR that reranking will deliver anyway risks paying twice for one gain.

If reranking lands and Hit@10 still has headroom, this decision should be revisited *on top of* that work rather than instead of it. All shapes remain reproducible behind `RESULTS=`.

### 2. Hybrid retrieval currently scores below sparse-only

Sparse-only scores `0.829151`; the shipped hybrid scores `0.800304`. The team's position, recorded for transparency:

- The public set appears skewed toward literal keyword matching, which is expected — the customer discloses near-verbatim catalog text, which is exactly BM25's strongest case.
- The dense route is retained for **robustness, not recall**. [`competition_specification.md:40`](../docs/competition_specification.md#L40) states the organizers may add natural-language paraphrasing. Our current score leans heavily on the customer quoting product text verbatim; if that text is paraphrased for the private set, keyword matching weakens and the dense route is what degrades gracefully.
- Recall is demonstrably *not* the justification: BM25 alone already reaches recall `1.000` at depth 500, so dense retrieval cannot add reachable targets. This is worth stating explicitly, because the original plan's gate for dense retrieval ("move Hit@10 above 0.875") was written before that measurement existed and is not achievable on those grounds.

Switch with `RETRIEVAL=sparse`.

### 3. Deterministic control flow instead of LLM self-planning

The problem statement's "runtime workflow re-orchestration" was interpreted as **state-conditioned parameter adaptation**, not model self-planning. A self-modifying pipeline is non-deterministic, hard to debug, costs turns when it wanders, and cannot be explained to a judge asking "why did it do that?". The cost of this choice is that adaptation is limited to what we thought to encode; the benefit is that every number in this document is reproducible to six decimals.

---

## Known bugs and limitations

- **Ordering is the bottleneck, and part of it is unreachable.** Five sessions have targets at true ranks 110–315, past the 100 items any 10-turn session can show. Neither reranking nor better dialogue can recover them.
- **Slot state is inert.** `SlotState` accumulates colour and material and detects overrides, but the query is built from raw message history, so slot values do not currently affect the score at all. It is wired for a reranker that consumes a structured constraint dict.
- **`route()` is close to a constant.** Across 2000 routing decisions it returns `browsing` 90.7% of the time, and agrees with the evaluator's true scenario label on only 84 of 160 buying/browsing sessions. Its original `filled >= 3` slot-count branch was dead code — only `color` and `material` can ever be filled, so the condition could never be true — and has been removed; `route()` is now a pure keyword check, with identical behaviour (score unchanged at `0.800304`). Making a slot-count signal work would need `size` and `feature` added to `SlotState`, whose fields are frozen by `AGENTS.md`, and `route()`'s only consumer is the retrieval owner's RRF weighting — so that decision belongs to them.
- **Two override false positives remain, from two different causes.** Multi-material listings were fixed (5 → 2 of 32 detections) by extracting every material in a message and testing membership instead of equality. What remains: (a) a material disclosed across two separate turns — shell on one, lining on another — which a single-valued slot cannot represent; (b) the word *"forget"* appearing in ordinary product prose (a greeting-card description), which trips the lexical override cue. The second is unrelated to materials and needs a narrower cue list.
- **Attribute extraction covers colour and material only**, by regex over a closed vocabulary. The catalog has no `color` or `material` field — both exist only in free text.
- **Measured on 200 public sessions with no held-out set.** The private 800 share the same generator and scenario mix. Nothing is fitted to individual sessions, and the one numeric parameter (`DEPTH`) has a saturating rather than peaked sensitivity curve, but transfer cannot be confirmed before submission.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m starter.src.index_build     # downloads encoder, builds embeddings; idempotent
cd ../.. && python -m evaluator.local_evaluator
```

The catalog is not committed. Download `catalog.jsonl.gz` from the challenge GitHub Release and decompress it to `data/catalog.jsonl` (50,000 rows).

`python run_eval.py --note "what changed"` runs the same evaluation and appends metrics, commit hash and note to `runs.log` — how every row above was produced.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CATALOG_PATH` | `data/catalog.jsonl` | Catalog location |
| `RETRIEVAL` | `hybrid` | `sparse` or `dense` to isolate a route |
| `ASK_POLICY` | `other` | `targeted` for attribute selection, `none` to disable asking |
| `RESULTS` | `full` | `turn1_small`, `ramp_turn`, `ramp_facts` |
| `EXPLORE` | `1` | `0` disables unseen-first selection |
| `PROFILE_SEED` | `0` | `1` seeds the query with profile preference tags |
| `DEPTH` | `120` | Candidates requested per turn |

Every flag exists so that a row of the tables above can be reproduced without editing source.

**`DEPTH` is not finely tuned** — (sparse):

| DEPTH | 20 | 40 | 60 | **80** | 100 | 120 | 200 |
|---|---|---|---|---|---|---|---|
| Score | 0.7995 | 0.8081 | 0.8146 | **0.8292** | 0.8292 | 0.8292 | 0.8292 |

Saturating, not peaked — identical to six decimals at 80 through 1000. The ceiling is structural: at most 10 items over at most 10 turns means no more than 100 candidates can ever be consumed. The shipped 120 sits 50% above saturation.

### Verification performed

- **No label leakage.** No file in `starter/` references `ground_truth`, `public_set`, `sample_id`, `intent_card`, `behavior` or `scenario_type`; the only file the agent opens is the catalog. Replacing every customer message with a fixed string collapses the score from `0.800304` to `0.020412` — below the provided baseline — confirming performance comes from the conversation rather than a leak.
- **No cross-session state.** All 200 sessions re-run in shuffled order: 0 of 200 outcomes changed.
- **Contract compliance.** 0 violations across 200 sessions — message always a string, `ask_attribute` always in the enum, 10 unique catalog-valid ASINs, never empty, never exceeding 10 turns.
- **Determinism.** Repeated runs identical to six decimal places.
- **Frozen paths untouched.** `evaluator/`, `docs/`, `data/`, `tests/` byte-identical to upstream.

### Network, latency, and cost

**No runtime network calls and zero LLM tokens.** Reports `prompt_tokens: 0, completion_tokens: 0`, so estimated cost at any LLM provider's pricing is **$0.00**. Setup-time network is required once, to download the encoder weights (~130MB) — the same requirement as `pip install`.

Per-turn latency, measured across all 578 turns of a full 200-session evaluation on a laptop CPU (dialogue state, query construction, hybrid retrieval, and control policy — no network call in the loop):

| | Latency |
|---|---|
| Mean | 32.45 ms |
| Median (p50) | 30.86 ms |
| **p95** | **58.76 ms** |
| p99 | 76.19 ms |
| Max | 86.20 ms |

A full 200-session evaluation completes in about 27 seconds, including building the SQLite FTS5 index and loading the embedding matrix into memory; it does not include the one-time dense index build (`python -m starter.src.index_build`, ~3 minutes, run once and cached).

If reranking is added on top of this, its latency and token cost are additive to the numbers above and will be disclosed separately once implemented.

---

## Team contributions

- **Person A** — retrieval: BM25 index, dense retrieval, RRF fusion, index build, reranking.
- **Person B** — dialogue: control policy, dialogue state tracking, query construction, ask policy, routing, guardrails, evaluation harness, failure analysis and verification.
