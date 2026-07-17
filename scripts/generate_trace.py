#!/usr/bin/env python3
"""Tier-2: regenerate the per-token entropy trace from the model itself.

This is the "regenerate the data, not just the figure" path. It runs the state
object interface on the wombat query and writes data/entropy_trace_wombat.json,
which scripts/fig1_entropy_trace.py then renders.

REQUIRES MODEL WEIGHTS (and realistically a GPU). The default figure path
(`make figures`) uses the committed trace and needs none of this. Exact
regeneration depends on the model build; see REPRODUCTION.md.

Usage:
    python scripts/generate_trace.py [--model MODEL_ID] [--max-tokens N]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUERY = "What shape is wombat scat?"
OUT = ROOT / "data" / "entropy_trace_wombat.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/olmo-3-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    from observability import StateObjectInterface

    interface = StateObjectInterface(args.model)
    try:
        records = interface.observe_tokens(QUERY, max_tokens=args.max_tokens)
    finally:
        interface.cleanup()

    out = {
        "query": QUERY,
        "expected": "cube",
        "category": "knowable_weird",
        "note": "Generated answer is a fabrication; wombat scat is distinctively cube-shaped.",
        "source_model": args.model,
        "source_trace": "regenerated via scripts/generate_trace.py",
        "tokens": [
            {"token_text": r["token_text"], "entropy": r["entropy"],
             "logprob": r["logprob"], "top5_mass": r["top5_mass"]}
            for r in records
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(records)} tokens from {args.model}")


if __name__ == "__main__":
    main()
