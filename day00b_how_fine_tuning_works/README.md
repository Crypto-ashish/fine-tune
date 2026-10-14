# Day 0B — How Fine-Tuning Works

What fine-tuning is, why it's done, when not to use it, and how it can help you create content in your own voice.

## Files

| File | What it is |
|---|---|
| `Day00B_How_Fine_Tuning_Works.docx` | The full lesson: the three layers of a model, the training loop, prompting vs RAG vs fine-tuning, content-creation use cases, toy demo results, key terms. |
| `toy_finetune.py` | A tiny word-predicting model in plain Python. It "pretrains" on general text, then "fine-tunes" on bakery captions, so you can watch its writing change. No GPU or installs needed. |
| `sample_caption_dataset.jsonl` | 6 synthetic Instagram-caption training examples showing the format of a real fine-tuning dataset. Replace them with your own posts later. |

## Run the toy demo

```
python toy_finetune.py
```

In Colab, upload the file and run `!python toy_finetune.py`. Then change `FINETUNE_EPOCHS` to 1 and 10 to see weak training and overfitting.

The toy learns by counting word pairs. Real models learn by adjusting weights with gradient descent (Days 6–8). The effect is the same idea: the model's output shifts toward your data.
