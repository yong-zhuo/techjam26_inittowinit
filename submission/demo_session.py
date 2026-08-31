from __future__ import annotations

import argparse
import json
import sys

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from submission.agent import Agent
from submission.src.retrieval import interface

# Replays one public session using the evaluator's own customer simulator.
_captured: dict = {}


def capture_retrieve() -> None:
    original = interface.retrieve

    def wrapped(query, slots, track, top_k):
        result = original(query, slots, track, top_k)
        _captured.update(ranked=list(result), track=track, query=query)
        return result

    interface.retrieve = wrapped


def replay(sample: dict, agent: Agent, catalog_ids: set[str], categories: dict, products: dict):
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    rows = []
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, 10)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        full = _captured.get("ranked", [])
        rows.append({
            "turn": turn,
            "user": user_message,
            "ask": response["ask_attribute"],
            "agent": response["message"],
            "track": _captured.get("track"),
            "candidate_rank": full.index(target) + 1 if target in full else None,
            "shown_rank": ranked.index(target) + 1 if target in ranked else None,
            "query_terms": len(set(_captured.get("query", "").lower().split())),
        })

        if (override_applied and target in ranked) or turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
            user_message = str(override.get("message", ""))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
    return target, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one public session turn by turn")
    parser.add_argument("--scenario", default="buying",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = [s for s in load_jsonl(args.dataset) if s["scenario_type"] == args.scenario]
    if not samples:
        sys.exit(f"no {args.scenario} sessions in {args.dataset}")
    catalog_ids, categories, products = catalog_index(args.catalog)

    capture_retrieve()
    sample = samples[args.index % len(samples)]
    target, rows = replay(sample, Agent(args.catalog), catalog_ids, categories, products)

    print(f"\n{sample['sample_id']}  ({sample['scenario_type']})")
    print(f"target {target}  {str(products[target].get('title'))[:70]}\n")
    for row in rows:
        rank = row["candidate_rank"] or "-"
        shown = row["shown_rank"] or "-"
        print(f"turn {row['turn']}  [{row['track']}]  candidate_rank={rank}  "
              f"shown={shown}  query_terms={row['query_terms']}")
        print(f"  customer: {row['user'][:160]}")
        print(f"  agent:    {row['agent']}  (ask_attribute={row['ask']!r})\n")
    print(json.dumps({"sample_id": sample["sample_id"], "target": target, "turns": rows}, indent=2))


if __name__ == "__main__":
    main()
