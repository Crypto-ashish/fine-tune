"""
Day 0C mini project: turn a messy raw JSONL file into clean train / validation files.

Steps: load (skip broken lines) -> clean text -> remove invalid records
       -> remove duplicates -> shuffle and split -> save -> report.

Run:   python prepare_dataset.py                      (uses raw_captions.jsonl)
       python prepare_dataset.py my_file.jsonl        (any file with the same keys)
Output: data/train.jsonl and data/val.jsonl
"""
import json
import os
import random

REQUIRED = ["instruction", "input", "output"]


def load_jsonl(path):
    """Read one JSON record per line. Skip blank or broken lines instead of crashing."""
    records = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipped line {number}: not valid JSON")
    return records


def clean_text(text):
    """Remove spaces at the ends and collapse repeated spaces inside."""
    return " ".join(str(text).split())


def find_problems(record):
    """Return a list of problems with one record. An empty list means it is valid."""
    problems = []
    for key in REQUIRED:
        if key not in record:
            problems.append(f"missing '{key}'")
        elif record[key] == "":
            problems.append(f"empty '{key}'")
    return problems


def save_jsonl(records, path):
    """Write records one per line, creating the folder if needed."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(raw_path, out_dir="data", val_fraction=0.2, seed=42):
    """Run the whole cleaning pipeline and return a report dictionary."""
    report = {}

    records = load_jsonl(raw_path)
    report["loaded"] = len(records)

    for record in records:
        for key in record:
            record[key] = clean_text(record[key])

    valid = [r for r in records if not find_problems(r)]
    report["invalid_removed"] = len(records) - len(valid)

    seen = set()
    deduped = []
    for r in valid:
        key = (r["instruction"].lower(), r["input"].lower(), r["output"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    report["duplicates_removed"] = len(valid) - len(deduped)

    shuffled = deduped[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val, train = shuffled[:n_val], shuffled[n_val:]

    save_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(out_dir, "val.jsonl"))
    report["train"] = len(train)
    report["val"] = len(val)
    return report


if __name__ == "__main__":
    import sys
    raw_file = sys.argv[1] if len(sys.argv) > 1 else "raw_captions.jsonl"
    result = prepare(raw_file)
    for key, value in result.items():
        print(f"{key}: {value}")
