"""
Day 0B - A toy "fine-tuning" you can run anywhere (no GPU, no libraries).

What it shows:
  1. "Pretrain" a tiny word-predicting model on general sentences.
  2. Ask it to write. It sounds generic.
  3. "Fine-tune" it on a few bakery Instagram captions.
  4. Ask it again. Same model, same prompt - it now writes in the bakery's style.

How this toy differs from a real LLM:
  This toy learns by COUNTING which word follows which. A real model like Qwen
  learns by adjusting millions of numbers (weights) with gradient descent
  (Days 6-8). But the idea is the same: the model learns "what usually comes
  next" from data, and fine-tuning shifts that toward YOUR data.

Run:  python toy_finetune.py
"""
import random

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
# General text - stands in for the internet-scale data a real model is pretrained on.
GENERAL_TEXT = [
    "our company makes fresh products for every customer .",
    "the new product is good for the customer .",
    "our team delivers good service every day .",
    "fresh ideas help the company grow .",
    "the customer wants a good price and fast delivery .",
    "our new office is open every day .",
    "the team makes a new plan for the product .",
]

# Your own data - stands in for your fine-tuning dataset (your best captions).
BAKERY_CAPTIONS = [
    "our gluten free cookies are baked fresh every morning in delhi .",
    "fresh from the oven : gluten free banana bread , soft and golden .",
    "our almond flour brownies are fudgy , gluten free and made with love .",
    "baked fresh every morning , no maida , no compromise .",
    "our new millet cookies are crunchy , wholesome and gluten free .",
]

FINETUNE_EPOCHS = 3   # how many times we go through the bakery captions (try 1, 3, 10)
SEED = 7              # fixes the randomness so you get the same output every run


# ---------------------------------------------------------------------------
# 2. THE "MODEL": a table of next-word counts
# ---------------------------------------------------------------------------
# model["our"] = {"company": 1, "team": 1, ...} means:
#   after the word "our", we have seen "company" once, "team" once, ...
def train(model, sentences, epochs=1):
    """Update the model's counts from a list of sentences. Returns the same model."""
    for _ in range(epochs):                         # repeat over the data
        for sentence in sentences:                  # every training example
            words = sentence.split()
            for current, nxt in zip(words, words[1:]):   # pairs: (word, next word)
                if current not in model:
                    model[current] = {}
                model[current][nxt] = model[current].get(nxt, 0) + 1
    return model


def next_word_probabilities(model, word):
    """Turn counts after `word` into percentages, highest first."""
    counts = model.get(word, {})
    total = sum(counts.values())
    probs = {w: c / total for w, c in counts.items()}
    return dict(sorted(probs.items(), key=lambda pair: pair[1], reverse=True))


def generate(model, start_word, max_new_words=12, seed=SEED):
    """Write text one word at a time - exactly what model.generate() does with tokens."""
    rng = random.Random(seed)
    words = [start_word]
    for _ in range(max_new_words):
        probs = next_word_probabilities(model, words[-1])
        if not probs:                               # nothing ever followed this word
            break
        choice = rng.choices(list(probs), weights=list(probs.values()))[0]
        words.append(choice)
        if choice == ".":                           # "." acts as our end-of-text token
            break
    return " ".join(words)


def show(model, title):
    print("=" * 64)
    print(title)
    print("-" * 64)
    for start in ["our", "fresh"]:
        top = next_word_probabilities(model, start)
        top3 = ", ".join(f"{w} {p:.0%}" for w, p in list(top.items())[:3])
        print(f"After '{start}', most likely next words: {top3}")
    print()
    for s in [1, 2, 3]:
        print(f"  Sample {s}: {generate(model, 'our', seed=SEED + s)}")
    print()


# ---------------------------------------------------------------------------
# 3. RUN THE EXPERIMENT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = train({}, GENERAL_TEXT)                       # "pretraining"
    show(model, "BEFORE fine-tuning (only general text)")

    model = train(model, BAKERY_CAPTIONS, epochs=FINETUNE_EPOCHS)   # "fine-tuning"
    show(model, f"AFTER fine-tuning on {len(BAKERY_CAPTIONS)} bakery captions x {FINETUNE_EPOCHS} epochs")

    print("Notice: the model still knows its general words (company, team...),")
    print("but its most likely choices have shifted toward YOUR style.")
