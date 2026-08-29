# PLAN.md

Roadmap, phase gates, and verification for the TechJam Track 4 submission. Read the phase you are about to start before starting it.

Part of a set: `AGENTS.md` (always-loaded rules and commands), `PLAN.md` (roadmap and gates), `DESIGN.md` (component rationale and research), `SUBMISSION.md` (packaging and rules).

---

## Scoring

```
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

- **Hit@10** — was the target in the returned top 10
- **MRR** — reciprocal of its rank (1st = 1.0, 5th = 0.2)
- **MTTC** — mean turn at which the target *first* appeared. A miss counts as turn 11

**Baseline:** `Hit@10 = 0.125`, `MRR = 0.068034`, `MTTC = 9.81`, score `0.10671`.

### The measurement that sets all priorities

The evaluator only reveals new information when the agent asks. `local_evaluator.py:170` — if `ask_attribute` is `None`, the simulated customer replies *"Ask me about one specific attribute"* and discloses nothing. The starter never asks, so it gets turn-1 information and then nine silent turns. `ask_attribute="other"` (`line 180`) bypasses the type filter and returns the next two undisclosed constraints of any kind.

Measured with a probe agent over the full 200 public sessions, **BM25 only** — no dense retrieval, no fusion, no LLM:

| Variant | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| baseline: never ask, message only | 0.1250 | 0.0680 | 9.81 | 0.1067 |
| never ask, full history | 0.2700 | 0.1514 | 8.60 | 0.2284 |
| ask `other`, message only | 0.5600 | 0.3992 | 6.14 | 0.4970 |
| cycle attributes, full history | 0.7950 | 0.5158 | 4.81 | 0.6760 |
| 3 targeted then `other`, full history | 0.8400 | **0.5651** | 4.40 | 0.7215 |
| **ask `other`, full history** | **0.8750** | 0.5400 | **3.46** | **0.7504** |

These are probe results, not the committed agent. The repo currently scores the baseline.

**Consequence.** Two trivial changes — keep a message history, and set `ask_attribute` — are worth roughly 7× the baseline score. Build them first, before any retrieval work.

### Where the headroom is after that

At a score of 0.7504:

| Component | Now | Max | Remaining |
|---|---|---|---|
| Hit@10 (50%) | 0.875 | 1.0 | +0.063 |
| **MRR (30%)** | 0.540 | 1.0 | **+0.138** |
| Efficiency (20%) | 0.754 | 1.0 | +0.049 |

**MRR becomes the largest lever**, which is what LLM semantic reranking targets. Dense retrieval still matters — it is a stated requirement and buys the remaining Hit@10 — but it is no longer the single dominant win it appeared to be against a baseline that never asked questions.

### Human judging

65% of the overall hackathon score is human-judged: Technical Execution 35%, Innovation 20%, Impact 20%, Feasibility 15%, Presentation 10%. Eight hours are reserved for README, tables, and video. Not negotiable.

---

## Files to build

Already done: `agent.py`, `src/dialog/state.py`, `src/retrieval/interface.py`, `src/retrieval/sparse.py`.

| File | Lines | Owner |
|---|---|---|
| `src/dialog/query.py` — rewriting + dual-track routing | ~35 | B |
| `src/index_build.py` — download and embed, idempotent | ~40 | A |
| `src/retrieval/dense.py` — cosine | ~40 | A |
| `src/retrieval/fusion.py` — RRF | ~15 | A |
| `src/retrieval/rerank.py` — listwise LLM + fallback | ~80 | A |
| `src/obs/cache.py` — disk cache | ~25 | A |

**Ownership.** A owns `src/retrieval/`, `index_build.py`, `cache.py`. B owns `agent.py`, `src/dialog/`. The seam is frozen: `retrieve(query, slots, track, top_k)`.

**Deliberately not built:** structured pre-filter, entropy clarification, faithful explanations, trace logging, unit tests, `eval_tools/` scripts. Reasons in `DESIGN.md`; the evaluator's per-scenario metrics are the regression signal.

---

## Phases

### Phase 1 — the cheap win (~4h each, parallel from the start)

| Person B — most of the score lives here | Person A — independent, start immediately |
|---|---|
| `agent.py`: keep a message history | `index_build.py`: `bge-small-en-v1.5`, pinned revision, idempotent |
| Ask policy as one swappable function | `dense.py`: brute-force cosine, instruction prefix. No FAISS |
| Guardrails: never-empty, no ask at turn 10, ASIN dedupe and validation | |

**Gate: score ≥ 0.70.** Run history and asking as separate eval runs so both are attributable.

### Phase 2 — hybrid retrieval and context (~5h each)

| Person B | Person A |
|---|---|
| `query.py`: history + slots → query | `fusion.py`: RRF, `k=60`, weighted by `track` |
| Dual-track routing, ~10 lines, same file | Measure BM25 vs dense vs fused on Hit@10 |
| Profile seeding from `user_profile`, behind a flag | Field ablation: title vs title + category |

Profile seeding rule: an explicitly stated constraint always beats a profile prior. The tags are generic (`fit`, `comfort`, `durability`) and may dilute the query — if it costs score, gate it off and report that.

**Gate:** dense must move Hit@10 above 0.875. If it does not, keep the simpler system and say so.

### Phase 3 — ranking and adaptation (~6h each)

| Person B | Person A |
|---|---|
| **Adaptive ask policy** — detect *"I don't have an additional preference"* and switch from targeted to `other` | `rerank.py`: listwise, one call per rerank, `temperature=0`, prompt with the slot dict not the transcript |
| Log every adaptive rule that fires | **Selective reranking**: skip turns that add no information, skip when RRF is already confident |
| `--no-llm` mode, must still beat baseline | RRF fallback, permanent switch on first failure |
| | `cache.py` — build before iterating on prompts |

Budget with selective reranking: ~500–800 calls per full run, ~150–200 on a 50-session dev subset.

**Gate:** MRR moves above 0.54. Zero crashed sessions. `--no-llm` still scores well.

### Phase 4 — deliverables (~8h each). 65% of the grade.

| Person A | Person B |
|---|---|
| README: method, model choice, limitations | Ablation table, from the logged runs |
| Network requirement: setup-time vs runtime | Scenario table, free from the evaluator |
| `requirements.txt` with exact pins, `setup.sh` | Failure analysis: read the 20 worst sessions |
| | Cost disclosure: latency mean and p95, tokens, estimated cost |
| **Both:** demo video, submission checklist, clean clone test | |

---

## Verification

The evaluator is the test suite. Run it after every change:

```bash
python run_eval.py --note "what changed"
```

Watch the four scenario rows, not just the aggregate. A broken component usually shows up in one scenario while the others hold steady — in aggregate it just looks like "could be better".

### Debugging order

Never tune the reranker while recall is broken.

| Symptom | Cause | Do not touch |
|---|---|---|
| Target never in the candidate set | Retrieval: embeddings, fusion, over-filtering | The rerank prompt |
| Hit@10 low, candidates fine | Reranker dropping the target | Retrieval |
| MRR low, Hit@10 fine | Reranker ordering: prompt, context | Retrieval |
| MTTC high, Hit@10 fine | Asking too late, or not at all | Anything else |
| One scenario far below others | That scenario's handler | The global pipeline |

**Invariants.** Hit@10 ≥ MRR, always. A violation is a bug.

### The three tables

**Ablation** — one row per component, each behind a flag:

| Config | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| BM25 baseline | 0.125 | 0.068 | 9.81 | 0.1067 |
| + message history | | | | |
| + clarification | | | | |
| + query rewriting | | | | |
| + dense retrieval | | | | |
| + RRF fusion | | | | |
| + LLM rerank | | | | |
| + adaptive ask | | | | |
| Full system | | | | |

Plus the two gated variants: profile seeding on/off, and ask policy `other` vs targeted-then-`other`.

Every row should move a number. A row that does not is a component that has not earned its place — cut it and say why. This is the strongest available evidence of deliberate decision-making, which Technical Execution rewards.

**Scenario** — the final system split four ways. Free: the evaluator already reports it.

**Failure analysis** — read the 20 worst sessions and categorise: target never retrieved, retrieved but buried, override corrupted state, ran out of turns. Then three sentences naming the dominant failure mode and what would fix it.

---

## Cut list

Cut in this order if you fall behind:

1. Adaptive ask policy → fixed `other`
2. Profile seeding
3. Dense retrieval and RRF → BM25 only
4. LLM reranking → RRF order

**Never cut:** message history, clarification, query rewriting, guardrails, the three tables, README, video, clean clone test.

The first two items on the never-cut list are worth 7× the baseline on their own. A submission with those, a clean ablation table and a sharp README beats one with nine components and no writeup.
