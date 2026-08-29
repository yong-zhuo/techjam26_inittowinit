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

**Baseline to beat:** `Hit@10 = 0.125`, `MRR = 0.068034`, `MTTC = 9.81`

### The arithmetic that drives all prioritization

Misses are assigned turn 11. Working backwards from the baseline:

```
0.875 × 11        = 9.625
9.81 - 9.625      = 0.185
0.185 / 0.125     = 1.48   ← average turn when it does hit
```

The starter already recommends from turn 1 and hits at turn ~1.5 when it hits at all. Its MTTC of 9.81 is driven almost entirely by an 87.5% miss rate, not by conversational inefficiency.

**Consequence: Efficiency is close to a linear function of Hit Rate.** Improving recall improves all three metrics simultaneously. There is no meaningful accuracy-versus-turn-count tradeoff. Roughly 70% of achievable score is retrieval quality.

**When unsure what to work on, work on retrieval.**

### Human judging

65% of the overall hackathon score is human-judged: Technical Execution 35%, Innovation 20%, Impact 20%, Feasibility 15%, Presentation 10%. Eight hours are reserved for README, tables, and video. Not negotiable.

---

---

## Phases

### Phase 0 — Setup (0–6h). Both together, do not split.

| Task | Why |
|---|---|
| Fork the kit, `uv sync`, Python 3.10+ | — |
| Verify the SHA256 checksum | A truncated catalog fails silently |
| Run the starter, confirm `0.125 / 0.068 / 9.81` | Different numbers mean a broken environment |
| Read `docs/evaluation_config.json` | Confirm score weights |
| Read `docs/agent_api_contract.json` | **Authoritative.** Response shape and `ask_attribute` enum |
| Confirm how the evaluator resolves the agent path | Decides `submission/` wiring. Do not edit evaluator code |
| **Open the catalog and inspect it** | Is `price` a float or `"$24.99 - $31.50"`? Is `color` a field or only in titles? |
| Check whether sessions carry scenario labels | Decides whether the scenario table is possible |
| Inspect the `user_profile` dict from `reset()` | Decides turn-1 strategy |
| Agree `SlotState` fields and `retrieve()` signature | Freeze them |
| **Write `.gitignore` before the first commit** | A committed `data/` needs history rewriting to remove |
| Write the three tests | Contract, turn cap, override |
| Write `make eval` | Runs the evaluator, appends `{git hash, three numbers, note}` to `runs.log` |
| Ask organizers what "network disabled" covers | Setup versus runtime. Webinar and Discord both available |

**Gate:** baseline reproduced exactly. Tests pass. Interface frozen.

**Why six hours.** Catalog inspection decides the slot design. If colour lives only in titles, Person A needs a keyword-extraction pass over 50,000 titles — work to schedule now, not discover at hour 30.

### Phase 1 — State and tooling (6–14h). Split here.

**Person B:** `SlotState` with accumulation and override; override tests; profile seeding; simple clarification (first unfilled attribute — the entropy version comes in Phase 4); control policy in `agent.py`; scenario evaluator; trace logging.

**Person A:** catalog loading and normalization; title attribute extraction if Phase 0 found colour/material are not fields; BM25 index; `retrieve()` shim using BM25 only so B can integrate against real code.

**Gate:** full loop runs end to end. Override test passes. Score at or near baseline — nothing is optimized yet.

### Phase 2 — Retrieval (14–28h). Where the score is.

**Person A:** embedding pipeline with instruction prefix; `index_build.py`; dense search; RRF; pre-filter with recall guard; recall@500 diagnostic.

**Person B:** query rewriting; intent routing; LLM disk cache.

**Gate: Hit@10 must clearly beat 0.125.** If not, stop and debug recall@500. Do not proceed to reranking — there is no point tuning a reranker over a candidate set the target never entered.

**Merge checkpoint.** Run the scenario table.

### Phase 3 — Reranking and guardrails (28–36h)

**Person A:** RankGPT listwise rerank; provider via `RERANK_MODEL`; RRF fallback with permanent switch on first failure.

**Person B:** the four guardrails; `--no-llm` mode.

**Gate:** MRR moved. Zero crashed sessions across 200 dev sessions. `--no-llm` beats baseline.

### Phase 4 — Differentiators (36–42h)

| Build | Time | Note |
|---|---|---|
| Entropy clarification | 3h | The differentiator. Organizers named it |
| Faithful explanations | 45m | Debugging + demo + covers their route |
| Adaptive parameters | 30m | Three if-statements. Covers Pillar III |
| Cross-encoder fallback | 1h | **Only if** the hour-36 ablation shows LLM rerank moves MRR a lot |

**Gate at 42h: feature freeze.** Nothing new after this line.

### Phase 5 — Deliverables (42–50h). 65% of the score.

| Task | Who | Time |
|---|---|---|
| Ablation table (six eval runs) | B | 1h |
| Scenario table | B | 1h if labels exist |
| Failure analysis (read 20 worst sessions) | B | 1h |
| Cost disclosure — latency mean/p95, tokens, est. cost | B | 30m |
| README | A | 2h |
| `setup.sh` + README setup section | A | 30m |
| Demo video (`rich` trace replay) | Both | 2h |
| Submission checklist pass | Both | 30m |
| **Clean clone test** | Both | 1h, at hour 48 |

---

---

## Verification

### Tests, day one

1. **Contract** — construct a response, validate against `docs/agent_api_contract.json`
2. **Turn cap** — all 200 sessions, assert zero violations
3. **Override** — scripted contradiction, assert the old slot is `None`
4. **Determinism** — same session twice, identical output (`temperature=0` + prompt-hash cache)
5. **Recall ceiling** — is the target in the top 500 *before* reranking? The upper bound on Hit@10

Test 5 is the most useful diagnostic in the project.

Test 3 is the one that matters most in practice: most bugs show up in the score, but a broken override does not — it just makes the number quietly worse while you debug the wrong file.

### Debugging order

Never tune the reranker while recall is broken.

| Symptom | Cause | Do not touch |
|---|---|---|
| Recall@500 low | Retrieval: embeddings, fusion, over-filtering | The rerank prompt |
| Hit@10 low, recall fine | Reranker dropping the target | Retrieval |
| MRR low, Hit@10 fine | Reranker ordering: prompt, context | Retrieval |
| MTTC high, Hit@10 fine | Asking too many questions | Anything else |
| One scenario far below others | That scenario's handler | The global pipeline |

**Invariants.** Hit@10 ≥ MRR, always. Recall@500 ≥ Hit@10, always. A violation is a bug.

### The three tables

**Ablation** — six eval runs behind feature flags:

| Config | Hit@10 | MRR | MTTC |
|---|---|---|---|
| BM25 baseline | 0.125 | 0.068 | 9.81 |
| + dense retrieval | | | |
| + RRF | | | |
| + query rewriting | | | |
| + LLM rerank | | | |
| Full system | | | |

Every row should move a number. A row that does not is a component that has not earned its place. This is the strongest available evidence of deliberate decision-making, which Technical Execution (35%) rewards.

If the cross-encoder was built, add a three-way rerank comparison. Reranking reorders the existing top 30 rather than finding new candidates, so **Hit@10 barely moves and MRR is the column that matters.** Reporting that a local cross-encoder came close to RankGPT at zero API cost is a more interesting finding than "we used an LLM."

**Scenario** — the final system split four ways. Not required by the rules; it is a debugging tool. In aggregate a broken override looks like "0.48, could be better." Split, it is a `0.11` beside three healthy numbers. If sessions carry no scenario label, substitute: grep traces for sessions where `slots_erased` fired and check whether those hit.

**Failure analysis** — read the 20 worst sessions and categorize: target never in top 500 (recall), retrieved but buried (ranking), override corrupted state (state bug), ran out of turns (control policy). Then three sentences naming the dominant failure mode and what would fix it. Everyone reports their best number; almost nobody reports where they break.

---

---

## Cut list

Cut in this order if you fall behind:

1. Adaptive parameters
2. Faithful explanations
3. Entropy clarification → fall back to first-unfilled-attribute
4. LLM reranking → RRF order

**Never cut:** dense retrieval, RRF, override handling, guardrails, the three tables, README, video, clean clone test.

A submission with dense retrieval, working override handling, a clean ablation table and a sharp README beats one with nine components and no writeup.
