"""Can this machine run Qwen2.5-VL-3B locally?

Loads the model in 4-bit (as in the course tutorial), then measures:
  1. Inference: VRAM and speed at a low and a high image resolution,
     a follow-up turn (conversational memory) and a bounding-box prompt.
  2. QLoRA training: peak VRAM for one forward + backward pass with
     LoRA adapters on the language model (vision tower frozen).

Usage:  python check_local_vlm.py
First run downloads the model (~7.5 GB) into the Hugging Face cache.
"""
import gc
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
GB = 1024**3
LOW_RES, HIGH_RES = 448 * 448, 1024 * 1024  # pixels; roughly 256 vs 1300 image tokens


def resize_to(img, n_pixels):
    """Rescale (up or down) to about n_pixels, keeping the aspect ratio."""
    w, h = img.size
    s = (n_pixels / (w * h)) ** 0.5
    return img.resize((max(28, round(w * s)), max(28, round(h * s))))


def user_turn(text, with_image=False):
    content = [{"type": "image"}] if with_image else []
    return {"role": "user", "content": content + [{"type": "text", "text": text}]}


def assistant_turn(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def generate(messages, images, max_new_tokens=128):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
    n_in = inputs.input_ids.shape[1]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t

    new_tokens = out[0, n_in:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"  input tokens {n_in:5d} | output tokens {len(new_tokens):4d} | "
          f"{dt:5.1f} s ({len(new_tokens) / dt:4.1f} tok/s) | "
          f"peak VRAM {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    return text


# --------------------------------------------------
# SETUP
# --------------------------------------------------

assert torch.cuda.is_available(), "No CUDA GPU found"
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {props.name} | total {total / GB:.2f} GB | free before loading {free / GB:.2f} GB")

sample = load_dataset("flaviagiammarino/vqa-rad", split="test")
idx = next((i for i, q in enumerate(sample["question"]) if "lung" in q.lower()), 0)
image = sample[idx]["image"].convert("RGB")
print(f"Test image: VQA-RAD test #{idx}, {image.size[0]}x{image.size[1]}, "
      f"Q: {sample[idx]['question']!r} A: {sample[idx]['answer']!r}")

print(f"\nLoading {MODEL} in 4-bit ...")
t = time.perf_counter()
processor = AutoProcessor.from_pretrained(MODEL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model.eval()
print(f"Loaded in {time.perf_counter() - t:.0f} s | "
      f"weights in VRAM {torch.cuda.memory_allocated() / GB:.2f} GB")

# --------------------------------------------------
# 1. INFERENCE
# --------------------------------------------------

question = "Describe the key imaging findings in this image."
for label, n_px in [("low res (448x448)", LOW_RES), ("high res (1024x1024)", HIGH_RES)]:
    print(f"\n[Inference, {label}]")
    img = resize_to(image, n_px)
    answer = generate([user_turn(question, with_image=True)], [img])
    print(f"  > {answer[:300]}")

print("\n[Follow-up turn, low res, with conversation history]")
img = resize_to(image, LOW_RES)
history = [user_turn(question, with_image=True), assistant_turn(answer)]
follow_up = ("The patient arrived in the emergency department with fever and "
             "shortness of breath for three days. Does that change your interpretation?")
print(f"  > {generate(history + [user_turn(follow_up)], [img])[:300]}")

print("\n[Bounding-box prompt, low res]")
grounding = ("Locate the most important abnormality. Output only JSON: "
             '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]')
print(f"  > {generate([user_turn(grounding, with_image=True)], [img])[:300]}")

# --------------------------------------------------
# 2. QLoRA TRAINING (one step, memory only)
# --------------------------------------------------

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    task_type="CAUSAL_LM",
))
model.train()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[QLoRA] trainable params: {n_train / 1e6:.1f} M (language model only, vision frozen)")

report = ("Findings: " + "Patchy airspace opacity in the right lower zone with air bronchograms. " * 12
          + "Impression: findings suggest pneumonia; correlate clinically.")
for label, n_px in [("low res (448x448)", LOW_RES), ("high res (1024x1024)", HIGH_RES)]:
    messages = [user_turn("Write an emergency department imaging report.", with_image=True),
                assistant_turn(report)]
    text = processor.apply_chat_template(messages, tokenize=False)
    batch = processor(text=[text], images=[resize_to(image, n_px)], return_tensors="pt").to("cuda")
    batch["labels"] = batch["input_ids"].clone()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        torch.cuda.synchronize()
        print(f"  {label}: seq len {batch['input_ids'].shape[1]} | loss {loss.item():.2f} | "
              f"{time.perf_counter() - t:.1f} s/step | "
              f"peak VRAM {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  {label}: OUT OF MEMORY")
    model.zero_grad(set_to_none=True)
    loss = batch = None

print("\nDone.")
