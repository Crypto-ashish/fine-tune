# Day 0C — Coding Practice for Fine-Tuning

This is the second programming lesson; do Day 0 first. It works through one realistic job: turning a messy file of Instagram-caption examples into a clean training set. Along the way it covers the Python that fine-tuning code relies on. No GPU needed. It takes about 3–4 hours.

## Files

| File | What it is |
|---|---|
| `day00c_coding_practice.ipynb` | 16 lessons as a Colab notebook, with real outputs already saved. Run it from top to bottom. |
| `Day00C_Coding_Practice.docx` | The same lessons to read, plus a cheat sheet and a bridge table to later course days. |
| `prepare_dataset.py` | The Lesson 16 pipeline as a reusable script. It loads, cleans, validates and de-duplicates the data, splits it, and saves the result. |
| `raw_captions.jsonl` | The messy practice dataset: 10 lines with a broken line, a duplicate, a missing field, an empty output and extra spaces. |

## Lessons

1. Files: saving and reading text
2. JSON
3. JSONL datasets
4. `try` / `except`
5. `.keys()`, `.get()` and `None`
6. Cleaning text with string methods
7. Validating records and filtering lists
8. Removing duplicates with a set
9. Dataset statistics, `sorted` and `lambda`
10. Shuffling and splitting into train / validation
11. Saving the clean data
12. Named arguments and `**` unpacking
13. Classes, objects and `@dataclass`
14. Padding a batch by hand (attention masks)
15. A loop that learns (loss, gradient, learning rate)
16. Mini project: the whole pipeline

## Run the script

```
python prepare_dataset.py                 # uses raw_captions.jsonl
python prepare_dataset.py my_file.jsonl   # your own data, same keys
```

The script writes `data/train.jsonl` and `data/val.jsonl`. With the practice file, the expected report is: loaded 9, invalid_removed 2, duplicates_removed 1, train 5, val 1. That was checked with a real run.
