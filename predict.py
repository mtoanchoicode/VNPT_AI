#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv

# -------------------------------------------------
# Imports (package layout)
# -------------------------------------------------
from src.router import classify_one
from src.RAG.RAG_answerer import solve_rag
from src.STEM.stem_module import solve_stem
from src.Reasoning.infer import solve_reasoning

# -------------------------------------------------
# Paths (BTC will mount private_test.json here)
# -------------------------------------------------
INPUT_PATH = "/code/private_test.json"
OUTPUT_PATH = "submission.csv"

# Allow overriding router model via environment variable
# Example: ROUTER_MODEL=large
ROUTER_MODEL = "large"


def normalize_answer(ans: str, n_choices: int) -> str:
    """
    Normalize model output to A/B/C/D...
    """
    if not ans:
        return "A"

    ans = str(ans).strip().upper()
    for ch in ans:
        if "A" <= ch <= "Z":
            idx = ord(ch) - ord("A")
            if 0 <= idx < max(1, n_choices):
                return ch
    return "A"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ Missing input file: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ private_test.json must be a JSON list")

    results = []

    for item in data:
        qid = str(item["qid"])
        question = (item["question"] or "").strip()
        choices = item.get("choices") or []
        if not isinstance(choices, list):
            choices = []

        # -------------------------
        # 1) Route
        # -------------------------
        label, subtype = classify_one(question, choices, model=ROUTER_MODEL)

        # -------------------------
        # 2) Solve
        # -------------------------
        if label == "RAG":
            answer = solve_rag(question, choices)
        elif label == "STEM":
            answer = solve_stem(question, choices)
        else:
            answer = solve_reasoning(question, choices, subtype=subtype)

        answer = normalize_answer(answer, len(choices))

        results.append({
            "qid": qid,
            "answer": answer
        })

    # -------------------------
    # Write submission.csv
    # -------------------------
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ submission.csv generated with {len(results)} rows")


if __name__ == "__main__":
    main()
