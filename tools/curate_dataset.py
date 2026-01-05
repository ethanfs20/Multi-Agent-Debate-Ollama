#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate MAD dataset.jsonl into dataset_clean.jsonl")
    ap.add_argument("--input", required=True, help="Path to dataset.jsonl")
    ap.add_argument("--output", default=None, help="Path to dataset_clean.jsonl (default next to input)")
    ap.add_argument("--allow-urls", action="store_true")
    ap.add_argument("--allow-search-links", action="store_true")
    ap.add_argument("--allow-refusals", action="store_true")
    ap.add_argument("--allow-constraint-leaks", action="store_true")
    ap.add_argument("--allow-wrong-task", action="store_true")
    args = ap.parse_args()

    inp = args.input
    outp = args.output or os.path.join(os.path.dirname(inp), "dataset_clean.jsonl")

    kept = 0
    dropped = 0

    with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            q = ((obj.get("meta") or {}).get("quality") or {})
            if q:
                if (not args.allow_wrong_task) and q.get("wrong_task"):
                    dropped += 1
                    continue
                if (not args.allow_constraint_leaks) and q.get("constraint_leak"):
                    dropped += 1
                    continue
                if (not args.allow_refusals) and q.get("refusal"):
                    dropped += 1
                    continue
                if (not args.allow_search_links) and q.get("contains_search_links"):
                    dropped += 1
                    continue
                if (not args.allow_urls) and q.get("contains_url"):
                    dropped += 1
                    continue
                if q.get("likely_truncated"):
                    dropped += 1
                    continue
                if q.get("off_topic"):
                    dropped += 1
                    continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")
    print(f"Wrote: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
