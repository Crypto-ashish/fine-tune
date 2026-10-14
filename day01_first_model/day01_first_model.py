"""
Day 1 - Run your first AI model.

What this file does:
  1. Checks which hardware is available (NVIDIA GPU, Intel GPU, Apple GPU, or CPU).
  2. Downloads a small pretrained chat model from Hugging Face.
  3. Sends it a few prompts and prints the answers.

Run it in Google Colab (recommended) or on your laptop:
    python day01_first_model.py
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS - bring in code that other people wrote, so we can use it.
# ---------------------------------------------------------------------------
import time                                    # used to measure how long generation takes
import torch                                   # PyTorch: the maths engine that runs the model
from transformers import AutoTokenizer, AutoModelForCausalLM   # Hugging Face loaders

# ---------------------------------------------------------------------------
# 2. SETTINGS - the values you are encouraged to change.
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"     # which model to download from huggingface.co

MAX_NEW_TOKENS = 120     # maximum length of each answer, counted in tokens (~0.75 words each)
DO_SAMPLE = True         # True = some randomness; False = always pick the most likely next token
TEMPERATURE = 0.7        # only used when DO_SAMPLE is True. Low = focused, high = creative/random
TOP_P = 0.9              # only used when DO_SAMPLE is True. Keep the top 90% probability tokens
SEED = 42                # fixes the randomness so you get the same answer each run

PROMPTS = [
    "Explain what a pretrained language model is in two sentences.",
    "Give me three tips for reducing the cost of an injection-moulded plastic part.",
    "Write a one-line motto for a small bakery.",
]

# ---------------------------------------------------------------------------
# 3. PICK HARDWARE - use the fastest device that is actually available.
# ---------------------------------------------------------------------------
def pick_device():
    """Return the name of the best available device as a text string."""
    if torch.cuda.is_available():                              # NVIDIA GPU (e.g. Colab T4)
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():     # Intel GPU (e.g. Arc 130V)
        return "xpu"
    if torch.backends.mps.is_available():                      # Apple Silicon GPU
        return "mps"
    return "cpu"                                               # fallback: works everywhere, slower


def main():
    device = pick_device()
    print(f"PyTorch version : {torch.__version__}")
    print(f"Using device    : {device}")
    if device == "cuda":
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # 4. LOAD THE TOKENIZER AND THE MODEL
    # -----------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} (first run downloads about 1 GB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",          # use the number format the model was saved in (bfloat16 here)
    )
    model = model.to(device)   # move the model's weights onto the chosen device
    model.eval()               # switch to "inference mode" (we are not training today)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params / 1e6:.0f} million")

    # -----------------------------------------------------------------------
    # 5. GENERATE AN ANSWER FOR EACH PROMPT
    # -----------------------------------------------------------------------
    for prompt in PROMPTS:
        # 5a. Wrap the prompt in the chat format this model was trained on.
        messages = [
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt},
        ]

        # 5b. Text -> token IDs (numbers). return_dict=True also gives the attention mask.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,   # adds the "assistant, your turn" marker at the end
            return_tensors="pt",          # "pt" = return PyTorch tensors
            return_dict=True,
        ).to(device)

        # 5c. Build the generation settings. Sampling settings only apply if DO_SAMPLE is True.
        gen_settings = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE}
        if DO_SAMPLE:
            gen_settings["temperature"] = TEMPERATURE
            gen_settings["top_p"] = TOP_P

        # 5d. Run the model. no_grad() tells PyTorch not to track gradients (saves memory).
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_settings)
        seconds = time.time() - start

        # 5e. The output contains the prompt tokens + new tokens. Keep only the new ones.
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_length:]

        # 5f. Token IDs -> readable text.
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("\n" + "=" * 70)
        print(f"PROMPT : {prompt}")
        print("-" * 70)
        print(answer.strip())
        print(f"[{len(new_tokens)} new tokens in {seconds:.1f}s]")


# Run main() only when this file is executed directly (not when imported by another file).
if __name__ == "__main__":
    main()
