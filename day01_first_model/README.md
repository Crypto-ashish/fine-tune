# Day 1 — Run Your First AI Model

Load a small pretrained chat model (Qwen2.5-0.5B-Instruct) and generate answers to prompts.

> New to programming? Do [Day 0](../day00_programming_basics/) first. Its Bridge table explains every line of this code.

## Files

| File | What it is |
|---|---|
| `day01_first_model.ipynb` | Colab notebook: hardware check, model loading, token peek, `ask()` function, three experiments. **Start here.** |
| `day01_first_model.py` | Same workflow as one plain script, settings at the top. Use on a laptop or to see the full program in one place. |
| `requirements.txt` | Python packages needed. |
| `Day01_Tutorial.docx` | Full line-by-line explanation, experiments, debugging guide. |

## Option 1 — Google Colab (recommended)

1. Go to https://colab.research.google.com → **Upload** → choose `day01_first_model.ipynb`.
2. `Runtime → Change runtime type → T4 GPU → Save`.
3. Run cells top to bottom with **Shift + Enter**.

To run the `.py` file in Colab instead: upload it via the folder icon on the left, then run a cell containing:

```
!pip install -q -U "transformers>=4.56" accelerate
!python day01_first_model.py
```

## Option 2 — Your laptop (Intel Arc 130V)

```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install "transformers>=4.56" accelerate
python day01_first_model.py
```

The script prints `Using device: xpu` if PyTorch can see the Arc GPU, otherwise `cpu`. Both work for this small model; CPU is just slower. Check the current Intel GPU install steps at https://pytorch.org/get-started/locally/ if the XPU install fails.

## Success looks like

- `Using device: cuda` (Colab GPU), `xpu` or `cpu`
- `Model loaded. Parameters: 494 million`
- Three prompts, each followed by a sensible answer and a `[N new tokens in X s]` line

## Hardware estimate

0.5B parameters × 2 bytes (bfloat16) ≈ 1 GB of weights. Inference needs roughly 1.5–2 GB of RAM/VRAM in total. Any Colab runtime handles this.
