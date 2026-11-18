#!/usr/bin/env python
# coding: utf-8

# # MultiCaRe Examine VLM: MedGemma emergency-imaging research pipeline
# 
# This notebook is a complete research pipeline for the BN4601A group-project **Examine** task. It keeps the useful foundations from `group_proj_vanda.ipynb` - the filtered `OpenMed/multicare-case-images`/MultiCaRe workflow, image inspection, and leakage-safe grouping - but replaces the placeholder LLaMA-Factory route with a Hugging Face/TRL multimodal supervised fine-tuning flow for `google/medgemma-4b-it`.
# 
# **Course-brief requirements separated from the current request**
# 
# - The briefing's project structure is a single medical specialty with the core tasks **examine, explain, report**, plus a creative task. This notebook scopes the implementation to **Emergency medicine / Examine** and leaves Explain/Report as future extensions.
# - The current request adds the concrete implementation target: MedGemma 4B IT, the filtered MultiCaRe cohort, backpropagation for **10 epochs**, validation every epoch, best-checkpoint selection, held-out evaluation, and a `.pth` artifact.
# - A case-report keyword filter is not a clinician diagnosis and this literature-derived dataset is not a live emergency-room feed. The pipeline therefore requires clinician-adjudicated image labels before training.
# - The model output is structured imaging evidence plus a clinician-review action suggestion. It must not be treated as an autonomous diagnosis, treatment order, or triage decision.
# 
# The final `.pth` produced by default is a **loadable LoRA adapter package** containing the trained tensors and the base-model/configuration metadata. A full merged 4B model can also be exported, but it is substantially larger and still requires the MedGemma base files/tokenizer/processor for normal deployment.

# In[ ]:


# Environment note
#
# Do not uninstall packages from the shared/user Python environment here. The
# previous version removed huggingface_hub and then upgraded packages into a
# partially-corrupted user site. Provision the environment in the PBS job (or
# activate the same virtualenv before starting this kernel), then run this
# notebook from the beginning.
print("Use the CUDA-enabled environment selected by the PBS job; no package changes are made here.")


# In[ ]:


# Optional package check. Prefer a pre-created virtualenv/conda environment on Vanda.
# This cell deliberately does not install torch: a CPU-only torch wheel would
# make torch.cuda.is_available() false even when PBS allocated an A40.
import importlib.metadata as _metadata

for package_name in ["torch", "transformers", "accelerate", "peft", "trl", "bitsandbytes"]:
    try:
        print(f"{package_name}: {_metadata.version(package_name)}")
    except _metadata.PackageNotFoundError:
        print(f"{package_name}: MISSING")


# In[3]:


# Reproducible environment and paths.
# Run this cell before importing the MultiCaRe filtering module.
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
configured_scratch = os.environ.get("VANDA_SCRATCH", "/scratch/<NUS-ID>")
SCRATCH_ROOT = Path(configured_scratch) if Path(configured_scratch).exists() else PROJECT_ROOT / "vanda_runs"
RUN_ROOT = SCRATCH_ROOT / "medgemma_examine_super_new"

from datetime import datetime

BASE_RUN_ROOT = Path(
    os.environ.get("GROUP_PROJ_BASE_RUN_ROOT", "/scratch/<NUS-ID>/medgemma_examine_super_new")
)

RUN_NAME = os.environ.get(
    "GROUP_PROJ_RUN_NAME",
    f"run_{os.environ.get('PBS_JOBID', datetime.now().strftime('%Y%m%d_%H%M%S')).split('.')[0]}"
)

RUN_ROOT = BASE_RUN_ROOT / RUN_NAME

HF_HOME = Path(
    os.environ.get("HF_HOME", str(SCRATCH_ROOT / "hf_cache"))
).expanduser().resolve()
PIP_CACHE_DIR = SCRATCH_ROOT / "pip_cache"
for folder in (RUN_ROOT, HF_HOME, PIP_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

required_filter_packages = ["numpy", "pandas", "pyarrow", "PIL", "huggingface_hub"]
missing = [name for name in required_filter_packages if importlib.util.find_spec(name) is None]
if missing:
    raise ImportError(
        f"Missing filtering dependencies: {missing}. Install them in the Vanda environment "
        "before rerunning this cell; do not put tokens in notebook source."
    )

print("Project root:", PROJECT_ROOT)
print("Run root:", RUN_ROOT)
print("HF cache:", HF_HOME)
print("Missing packages:", missing)


# In[ ]:


# PBS/VANDA CUDA diagnostics. Define this before training; do not force
# CUDA_VISIBLE_DEVICES inside Python because PBS controls device isolation.
import os
import shutil
import socket
import subprocess
import sys


def gpu_preflight(torch_module, require_gpu=True):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print("Host:", socket.gethostname())
    print("Python:", sys.executable)
    print("torch:", getattr(torch_module, "__version__", "unknown"))
    print("torch file:", getattr(torch_module, "__file__", "unknown"))
    print("torch CUDA build:", getattr(torch_module.version, "cuda", None))
    print("CUDA_VISIBLE_DEVICES:", visible)
    print("PBS_JOBID:", os.environ.get("PBS_JOBID", "<unset>"))
    print("PBS_NODEFILE:", os.environ.get("PBS_NODEFILE", "<unset>"))

    nvidia_smi = shutil.which("nvidia-smi") if "shutil" in globals() else None
    if nvidia_smi:
        try:
            smi = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())
        except Exception as exc:
            print("nvidia-smi check failed:", repr(exc))
    else:
        print("nvidia-smi: not found in PATH")

    if not torch_module.cuda.is_available():
        reasons = []
        if not os.environ.get("PBS_JOBID"):
            reasons.append("this kernel does not appear to be running inside a PBS job")
        if visible == "":
            reasons.append("CUDA_VISIBLE_DEVICES is empty, so the scheduler hid all GPUs")
        if getattr(torch_module.version, "cuda", None) is None:
            reasons.append("the imported PyTorch build is CPU-only")
        if not reasons:
            reasons.append("the CUDA runtime/driver cannot be initialized by this Python environment")
        message = (
            "CUDA is unavailable in this Python process. " + "; ".join(reasons) + ".\n"
            "Run `hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
            "python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'` "
            "inside the same PBS job and use that Python executable for the notebook kernel."
        )
        if require_gpu:
            raise RuntimeError(message)
        print("WARNING:", message)
        return False

    try:
        torch_module.cuda.init()
        print("CUDA device count:", torch_module.cuda.device_count())
        print("CUDA device:", torch_module.cuda.get_device_name(0))
        print("Compute capability:", torch_module.cuda.get_device_capability(0))
        print("BF16 supported:", bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()))
    except Exception as exc:
        raise RuntimeError("PyTorch sees CUDA but failed to initialize the allocated GPU.") from exc
    return True


print("GPU preflight defined. It will run when training or evaluation starts.")


# In[4]:


# Reuse the project filter from group_proj_vanda.ipynb.
filter_candidates = [
    PROJECT_ROOT / "multicare_filter_colab" / "multicare_filter" / "src",
    PROJECT_ROOT / "multicare_filter" / "src",
    PROJECT_ROOT / "src",
]
for parent in PROJECT_ROOT.parents:
    filter_candidates.extend([
        parent / "multicare_filter_colab" / "multicare_filter" / "src",
        parent / "multicare_filter" / "src",
    ])
FILTER_SRC = next((p for p in filter_candidates if (p / "multicare_filter.py").is_file()), None)
if FILTER_SRC is None:
    raise FileNotFoundError("Cannot find multicare_filter.py. Keep multicare_filter_colab beside this notebook.")

sys.path.insert(0, str(FILTER_SRC))
import multicare_filter as mf

print("Filter module:", FILTER_SRC / "multicare_filter.py")


# ## 1. Filter the MultiCaRe emergency-imaging cohort
# 
# The filter identifies candidate emergency-related images and cases. It is a **cohort selector only**: keyword-derived department/vital fields are not image-level ground truth. The model target is built only from later clinician-reviewed annotations.

# In[5]:


CONFIG = dict(mf.CONFIG)
# Adjust only after inspecting the funnel. The default keeps the earlier project settings.
CONFIG


# In[6]:


img = mf.load_images()
cases = mf.annotate_cases(mf.load_cases())
selected, funnel = mf.apply_filters(img, cases, CONFIG)

print(f"Image records loaded: {len(img):,}")
print(f"Case records loaded: {len(cases):,}")
print(f"Selected image records: {len(selected):,}")
print("Filter funnel:")
print(funnel.to_string())
print("Selected image summary:")
print(
    selected[["modality", "radiology_region", "department_guess"]]
    .value_counts()
    .head(20)
    .rename("images")
    .to_frame()
    .to_string()
)


# In[7]:


import matplotlib.pyplot as plt

TARGET_MODALITY = "x_ray"     # Start with one modality for a coherent Examine task.
TARGET_REGION = None           # e.g. "chest"; set after inspecting the counts above.
PROJECT_SPECIALTY = "Emergency medicine"

candidate_images = selected[selected.modality.eq(TARGET_MODALITY)].copy()
if TARGET_REGION:
    candidate_images = candidate_images[
        candidate_images.radiology_region.fillna("").str.contains(TARGET_REGION, case=False, regex=False)
    ]
candidate_images = candidate_images.drop_duplicates("file_id").reset_index(drop=True)
if candidate_images.empty:
    raise ValueError("No candidate images. Inspect the modality/region table and adjust the pilot scope.")

print(f"Candidate images: {len(candidate_images):,}")
sample_rows = candidate_images.sample(min(12, len(candidate_images)), random_state=SEED)
sample_bytes = mf.load_image_bytes(sample_rows.file_id.tolist())
fig, axes = plt.subplots((len(sample_rows) + 3) // 4, 4, figsize=(14, 3.5 * ((len(sample_rows) + 3) // 4)), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, (_, row) in zip(axes.flat, sample_rows.iterrows()):
    ax.imshow(sample_bytes[row.file_id].convert("RGB"), cmap="gray")
    ax.set_title(f"{row.file_id} | {row.radiology_region}", fontsize=8, loc="left")
plt.tight_layout()
plt.show()


# ## 2. Create weakly supervised Examine labels for the college prototype
# 
# For a real clinical study, image-level labels must be clinician-adjudicated. For this classroom prototype, the default `weak_supervision` mode creates provisional labels automatically from the dataset's existing image captions and case metadata, so no external reviewer is required to demonstrate the end-to-end pipeline.
# 
# The intended target is a structured, image-grounded response:
# 
# `normality`, `verified_findings`, `visual_characteristics`, `anatomy`, `location_description`, `criticality`, `immediate_action_for_clinician_review`, `action_rationale`, `uncertainty`, `red_flags`, and optional normalized `boxes_xyxy_normalized`.
# 
# These are noisy proxy targets, not verified diagnoses or treatment recommendations. The notebook marks them as weak labels and keeps the prototype explicitly out of clinical deployment. Set `LABEL_MODE = 'clinician'` only when validated labels are available.

# In[8]:


OUTPUT_DIR = RUN_ROOT / "data"
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MODE = "weak_supervision"  # classroom demo; alternative: "clinician"
ANNOTATION_FILE = OUTPUT_DIR / ("weak_annotations.csv" if LABEL_MODE == "weak_supervision" else "clinician_annotations.csv")
ANNOTATION_COLUMNS = [
    "file_id", "article_id", "review_status", "image_quality", "normality",
    "verified_findings", "visual_characteristics", "anatomy", "location_description",
    "criticality", "immediate_action", "action_rationale", "uncertainty", "red_flags",
    "plain_language_explanation", "clinical_considerations", "report_impression",
    "clinician_followup", "boxes_xyxy_normalized", "reviewer_code", "adjudicated", "review_notes",
]

if LABEL_MODE == "weak_supervision":
    # Transparent proxy labels: source caption/case text are supervision signals, not ground truth.
    red_flag_terms = ["pneumothorax", "haemorrhage", "hemorrhage", "fracture", "dislocation", "occlusion", "perforation", "tamponade", "embol", "sepsis"]
    def weak_text(row, column, fallback):
        value = row.get(column, "")
        return fallback if pd.isna(value) or not str(value).strip() else str(value).strip()
    weak_rows = []
    for _, row in candidate_images.iterrows():
        caption = weak_text(row, "caption", "No source caption available.")
        case_text = weak_text(row, "case_text", "")
        source_text = f"{caption} {case_text}".casefold()
        red_flags = sorted({term for term in red_flag_terms if term in source_text})
        criticality = "emergent" if red_flags else "uncertain"
        region = weak_text(row, "radiology_region_granular", weak_text(row, "radiology_region", "unspecified"))
        weak_rows.append({
            "file_id": str(row.file_id), "article_id": str(row.article_id),
            "review_status": "weak", "image_quality": "unverified",
            "normality": "indeterminate", "verified_findings": caption,
            "visual_characteristics": caption, "anatomy": region,
            "location_description": region, "criticality": criticality,
            "immediate_action": "Prompt clinician review; correlate with history, examination, and vital signs before any action.",
            "action_rationale": "Weak proxy generated from source caption/case metadata; not clinically adjudicated.",
            "uncertainty": "high: weakly supervised label requiring clinical verification",
            "red_flags": ", ".join(red_flags) if red_flags else "none identified by keyword proxy",
            "plain_language_explanation": caption,
            "clinical_considerations": "Do not treat this proxy label as a diagnosis or treatment instruction.",
            "report_impression": f"Provisional caption-derived impression: {caption}",
            "clinician_followup": "Independent clinician/radiologist verification required.",
            "boxes_xyxy_normalized": "[]", "reviewer_code": "weak_supervision",
            "adjudicated": "synthetic", "review_notes": "Automatically generated for classroom demonstration only.",
        })
    annotations = pd.DataFrame(weak_rows, columns=ANNOTATION_COLUMNS)
    annotations.to_csv(ANNOTATION_FILE, index=False, encoding="utf-8-sig")
    print(f"Created {len(annotations):,} weakly supervised rows: {ANNOTATION_FILE}")
else:
    if not ANNOTATION_FILE.exists():
        template = candidate_images[["file_id", "article_id"]].copy()
        template["file_id"] = template.file_id.astype(str)
        for column in ANNOTATION_COLUMNS:
            if column not in template.columns:
                template[column] = ""
        template[ANNOTATION_COLUMNS].to_csv(ANNOTATION_FILE, index=False, encoding="utf-8-sig")
        print("Created clinician annotation template:", ANNOTATION_FILE)
    else:
        annotations = pd.read_csv(ANNOTATION_FILE, dtype={"file_id": str})
        missing_columns = [c for c in ANNOTATION_COLUMNS if c not in annotations.columns]
        if missing_columns:
            raise ValueError(f"Annotation CSV is missing columns: {missing_columns}")
        print("Existing clinician annotation file preserved:", ANNOTATION_FILE)


# In[9]:


# Validate annotations and assign article-level train/validation/test groups.
if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_FILE}")
annotations = pd.read_csv(ANNOTATION_FILE, dtype={"file_id": str})

if LABEL_MODE == "weak_supervision":
    reviewed = annotations.copy()
else:
    yes = lambda s: s.fillna("").astype(str).str.strip().str.casefold().eq("yes")
    reviewed = annotations[yes(annotations.review_status) & yes(annotations.adjudicated)
                           & annotations.image_quality.fillna("").astype(str).str.strip().str.casefold().eq("adequate")].copy()
target_columns = [
    "normality", "verified_findings", "visual_characteristics", "anatomy",
    "location_description", "criticality", "immediate_action", "action_rationale",
    "uncertainty", "red_flags", "plain_language_explanation", "clinical_considerations",
    "report_impression", "clinician_followup",
]
if reviewed.empty:
    raise ValueError("No usable rows found. Check candidate_images and the selected label mode.")
if reviewed.file_id.duplicated().any():
    raise ValueError("Each file_id must have exactly one adjudicated row.")
blank = reviewed[target_columns].isna() | reviewed[target_columns].astype(str).apply(lambda c: c.str.strip().eq(""))
if blank.any(axis=None):
    raise ValueError("Every target field must be completed; use 'none' or 'not applicable' explicitly.")

candidate_lookup = candidate_images[["file_id", "article_id"]].copy()
candidate_lookup["file_id"] = candidate_lookup.file_id.astype(str)
reviewed = reviewed.merge(candidate_lookup, on="file_id", how="inner", suffixes=("_annotation", "_candidate"))
if reviewed.empty:
    raise ValueError("No adjudicated file IDs match the current candidate cohort.")
reviewed["group_id"] = reviewed.article_id_candidate.fillna(reviewed.article_id_annotation).astype(str)

groups = np.array(sorted(reviewed.group_id.unique()))
if len(groups) < 3:
    raise ValueError("At least three distinct article_id groups are needed for train/validation/test.")
rng = np.random.default_rng(SEED)
rng.shuffle(groups)
n_test = max(1, round(0.15 * len(groups)))
n_val = max(1, round(0.15 * len(groups)))
if n_test + n_val >= len(groups):
    raise ValueError("Too few article groups for non-empty train/validation/test splits.")
test_groups = set(groups[:n_test])
val_groups = set(groups[n_test:n_test + n_val])
reviewed["split"] = reviewed.group_id.map(
    lambda g: "test" if g in test_groups else ("validation" if g in val_groups else "train")
)

# Rapid Vanda run: cap the materialized cohort while preserving the existing
# article-level split assignment. The cap is controlled by the PBS environment
# and defaults to 500 total images (400/50/50 across the splits).
MAX_DATASET_SAMPLES = int(os.environ.get("GROUP_PROJ_MAX_SAMPLES", "500"))
if MAX_DATASET_SAMPLES > 0 and len(reviewed) > MAX_DATASET_SAMPLES:
    split_targets = {
        "train": max(1, round(MAX_DATASET_SAMPLES * 0.80)),
        "validation": max(1, round(MAX_DATASET_SAMPLES * 0.10)),
        "test": max(1, MAX_DATASET_SAMPLES - round(MAX_DATASET_SAMPLES * 0.80)
                  - round(MAX_DATASET_SAMPLES * 0.10)),
    }
    sampled_parts = []
    for split_name, target in split_targets.items():
        split_rows = reviewed[reviewed.split.eq(split_name)]
        if split_rows.empty:
            continue
        sampled_parts.append(split_rows.sample(
            n=min(target, len(split_rows)), random_state=SEED
        ))
    reviewed = pd.concat(sampled_parts, ignore_index=True)
    print("Rapid-run sample cap:", MAX_DATASET_SAMPLES)
    print(reviewed.groupby("split").size().rename("sampled_images").to_string())
print(
    reviewed.groupby("split")
    .agg(images=("file_id", "count"), groups=("group_id", "nunique"))
    .to_string()
)
print(pd.crosstab(reviewed.split, reviewed.criticality).to_string())


# ## 3. Materialize images and build the Examine instruction dataset
# 
# Images are downloaded only for adjudicated rows. The response is assembled from clinician-entered fields, never from the unverified case report. Article-level grouping prevents images from the same publication appearing in both training and evaluation.

# In[10]:


import json
from PIL import Image

TASK_PROMPT = (
    "Examine this emergency-imaging image and return ONLY valid JSON with these keys: "
    "normality, observations, suspected_findings, anatomy, location, criticality, "
    "immediate_action_for_clinician_review, action_rationale, uncertainty, red_flags, "
    "and boxes_xyxy_normalized. Separate directly visible observations from clinical "
    "considerations. Do not claim certainty beyond the image. The action is a suggestion "
    "for clinician review, not an autonomous order."
)

def parse_boxes(value, file_id):
    raw = "" if pd.isna(value) else str(value).strip()
    boxes = [] if not raw else json.loads(raw)
    if not isinstance(boxes, list):
        raise ValueError(f"boxes_xyxy_normalized must be a list for {file_id}")
    for box in boxes:
        if not (isinstance(box, list) and len(box) == 4 and
                all(isinstance(v, (int, float)) for v in box) and
                0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Invalid normalized box for {file_id}: {box}")
    return boxes

def text_value(row, key):
    value = row.get(key, "")
    return "none" if pd.isna(value) or not str(value).strip() else str(value).strip()

manifest_rows = []
for start in range(0, len(reviewed), 16):
    batch = reviewed.iloc[start:start + 16]
    pil_by_id = mf.load_image_bytes(batch.file_id.tolist())
    for _, row in batch.iterrows():
        file_id = str(row.file_id)
        image_path = IMAGE_DIR / f"{file_id}.jpg"
        pil_by_id[file_id].convert("RGB").save(image_path, quality=95)
        boxes = parse_boxes(row.get("boxes_xyxy_normalized", ""), file_id)
        target = {
            "normality": text_value(row, "normality"),
            "observations": text_value(row, "visual_characteristics"),
            "suspected_findings": text_value(row, "verified_findings"),
            "anatomy": text_value(row, "anatomy"),
            "location": text_value(row, "location_description"),
            "criticality": text_value(row, "criticality"),
            "immediate_action_for_clinician_review": text_value(row, "immediate_action"),
            "action_rationale": text_value(row, "action_rationale"),
            "uncertainty": text_value(row, "uncertainty"),
            "red_flags": text_value(row, "red_flags"),
            "boxes_xyxy_normalized": boxes,
        }
        manifest_rows.append({
            "file_id": file_id,
            "group_id": row.group_id,
            "split": row.split,
            "image_path": str(image_path),
            "instruction": TASK_PROMPT,
            "response": json.dumps(target, ensure_ascii=False),
        })

manifest = pd.DataFrame(manifest_rows)
MANIFEST_FILE = OUTPUT_DIR / "examine_manifest.csv"
manifest.to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")
for split_name in ["train", "validation", "test"]:
    manifest[manifest.split.eq(split_name)].to_json(
        OUTPUT_DIR / f"{split_name}.jsonl", orient="records", lines=True, force_ascii=False
    )
print(manifest.groupby("split").size().rename("records").to_frame().to_string())
print("Manifest:", MANIFEST_FILE)


# ## 4. Configure MedGemma 4B IT and QLoRA
# 
# The implementation follows Google's published Hugging Face/TRL fine-tuning pattern: `AutoModelForImageTextToText`, `AutoProcessor`, a custom image/text collator, and PEFT LoRA/QLoRA. The official example requires a GPU with bfloat16 support and at least about 40 GB memory for its demonstrated setup; check Vanda's actual allocation before starting.
# 
# The switches are deliberately off until the dataset has passed clinician review and the GPU environment has been checked.

# In[11]:


MODEL_ID = "google/medgemma-4b-it"
USE_FAST_PROCESSOR = os.environ.get("MEDGEMMA_USE_FAST_PROCESSOR", "1") == "1"
TRAIN_EPOCHS = int(os.environ.get("GROUP_PROJ_EPOCHS", "1"))
RUN_TRAINING = True
RUN_HELDOUT_EVALUATION = os.environ.get("GROUP_PROJ_RUN_TEST", "1") == "1"
SAVE_FULL_MERGED_PTH = os.environ.get("GROUP_PROJ_SAVE_FULL_PTH", "0") == "1"

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 768
OUTPUT_DIR = RUN_ROOT / "checkpoints"
BEST_MODEL_DIR = RUN_ROOT / "best_model"
ADAPTER_PTH = RUN_ROOT / "medgemma_examine_best_adapter.pth"
FULL_PTH = RUN_ROOT / "medgemma_examine_best_merged.pth"
ARTIFACT_DIR = RUN_ROOT / "model_artifacts"
HF_ARTIFACT_DIR = ARTIFACT_DIR / "huggingface_adapter"
ARTIFACT_MANIFEST = ARTIFACT_DIR / "artifact_manifest.json"

print({
    "model": MODEL_ID,
    "epochs": TRAIN_EPOCHS,
    "backpropagation": "SFTTrainer",
    "best_selection": "lowest eval_loss after every epoch",
    "adapter_artifact": ADAPTER_PTH,
})


# In[ ]:


# The following loader is intentionally called only when model setup/training/evaluation is enabled.
def make_model_kwargs(torch):
    gpu_preflight(torch, require_gpu=True)
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Use a bfloat16-capable GPU (compute capability >= 8) for this QLoRA recipe.")
    from transformers import BitsAndBytesConfig
    dtype = torch.bfloat16
    return {
        "attn_implementation": "eager",
        "torch_dtype": dtype,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        ),
    }

def build_chat_dataset(frame):
    from datasets import Dataset, Image as HFImage
    ds = Dataset.from_pandas(frame[["image_path", "instruction", "response"]], preserve_index=False)
    ds = ds.cast_column("image_path", HFImage()).rename_column("image_path", "image")
    def add_messages(row):
        row["messages"] = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": row["instruction"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": str(row["response"])},
            ]},
        ]
        return row
    return ds.map(add_messages)

def make_collator(processor):
    def collate_fn(examples):
        texts, images = [], []
        for example in examples:
            images.append([example["image"].convert("RGB")])
            texts.append(processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            ).strip())
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        special_map = processor.tokenizer.special_tokens_map
        boi = special_map.get("boi_token")
        if boi is not None:
            labels[labels == processor.tokenizer.convert_tokens_to_ids(boi)] = -100
        labels[labels == 262144] = -100  # image placeholder token used by Gemma 3
        batch["labels"] = labels
        return batch
    return collate_fn

print("Model utilities defined.")


# ## 5. Train for 10 epochs with validation and best-model restoration
# 
# This is the only cell that launches backpropagation. It evaluates at the end of every epoch, saves an epoch checkpoint, and restores the checkpoint with the lowest validation loss. The test split is never used for model selection.

# In[ ]:


if RUN_TRAINING:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    gpu_preflight(torch, require_gpu=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = build_chat_dataset(manifest[manifest.split.eq("train")].reset_index(drop=True))
    val_ds = build_chat_dataset(manifest[manifest.split.eq("validation")].reset_index(drop=True))
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **make_model_kwargs(torch))
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            use_fast=USE_FAST_PROCESSOR,
        )
        print("Image processor fast mode:", USE_FAST_PROCESSOR)
    except (TypeError, ValueError) as exc:
        if not USE_FAST_PROCESSOR:
            raise
        print("Fast image processor unavailable; falling back to slow processor:", repr(exc))
        processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "right"
    model.config.use_cache = False

    lora_hparams = {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
        "modules_to_save": ["lm_head", "embed_tokens"],
    }
    peft_config = LoraConfig(**lora_hparams)

    # Keep this run independent of TensorBoard. Hugging Face Trainer's normal
    # tqdm progress bars remain enabled, while this callback stores the scalar
    # history needed for reproducible Matplotlib plots.
    class MatplotlibHistoryCallback(TrainerCallback):
        def __init__(self):
            self.records = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                record = {"step": int(state.global_step)}
                if state.epoch is not None:
                    record["epoch"] = float(state.epoch)
                record.update({key: float(value) for key, value in logs.items()
                               if isinstance(value, (int, float))})
                self.records.append(record)

    history_callback = MatplotlibHistoryCallback()
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to="none",
        disable_tqdm=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=make_collator(processor),
        callbacks=[history_callback],
    )
    train_result = trainer.train()

    # Save scalar history and loss/learning-rate plots beside the project
    # source, so they remain easy to retrieve after PBS stage-out.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR = PROJECT_ROOT / "training_plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_callback.records)
    history_df.to_csv(PLOTS_DIR / "training_history.csv", index=False)

    if not history_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        if "loss" in history_df:
            train_logs = history_df.dropna(subset=["loss"])
            axes[0].plot(train_logs["step"], train_logs["loss"], label="Training loss")
        if "eval_loss" in history_df:
            eval_logs = history_df.dropna(subset=["eval_loss"])
            axes[0].plot(eval_logs["step"], eval_logs["eval_loss"], "o-", label="Validation loss")
        axes[0].set_title("Training and validation loss")
        axes[0].set_xlabel("Optimizer step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        if "learning_rate" in history_df:
            lr_logs = history_df.dropna(subset=["learning_rate"])
            axes[1].plot(lr_logs["step"], lr_logs["learning_rate"])
        axes[1].set_title("Learning-rate schedule")
        axes[1].set_xlabel("Optimizer step")
        axes[1].set_ylabel("Learning rate")
        axes[1].grid(alpha=0.3)
        fig.savefig(PLOTS_DIR / "training_validation_curves.png", dpi=160)
        plt.close(fig)
    print("Saved training history and plots:", PLOTS_DIR)

    trainer.save_model(str(BEST_MODEL_DIR))
    processor.save_pretrained(str(BEST_MODEL_DIR))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve a self-contained copy of the PEFT/Hugging Face best adapter.
    # trainer.save_model writes adapter_model.safetensors and adapter_config.json.
    shutil.copytree(BEST_MODEL_DIR, HF_ARTIFACT_DIR, dirs_exist_ok=True)
    print("Training complete.")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best eval loss:", trainer.state.best_metric)
else:
    print("RUN_TRAINING=False. Review the manifest, annotation quality, and GPU allocation before enabling it.")


# ## 6. Package the best model as `.pth`
# 
# A 4B model's most practical `.pth` deliverable is a LoRA adapter package: it contains the learned adapter state, base-model identifier, LoRA configuration, processor identifier, and best-validation metadata. The loader in the evaluation section reconstructs MedGemma and applies this package. `SAVE_FULL_MERGED_PTH=True` is provided for a very large standalone state-dict export, but the Hugging Face directory remains the recommended deployment artifact.

# In[ ]:


def read_adapter_state(checkpoint_dir, torch):
    from safetensors.torch import load_file
    safe_file = Path(checkpoint_dir) / "adapter_model.safetensors"
    bin_file = Path(checkpoint_dir) / "adapter_model.bin"
    if safe_file.exists():
        return load_file(str(safe_file), device="cpu")
    if bin_file.exists():
        return torch.load(bin_file, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found under {checkpoint_dir}")

if RUN_TRAINING:
    import torch
    best_checkpoint = Path(trainer.state.best_model_checkpoint or BEST_MODEL_DIR)
    adapter_state = read_adapter_state(best_checkpoint, torch)
    package = {
        "artifact_type": "medgemma-4b-it-lora-adapter",
        "base_model": MODEL_ID,
        "processor": MODEL_ID,
        "best_checkpoint": str(best_checkpoint),
        "best_eval_loss": float(trainer.state.best_metric),
        "epochs_requested": TRAIN_EPOCHS,
        "task": "Emergency medicine Examine",
        "lora_hparams": lora_hparams,
        "state_dict": adapter_state,
        "warning": "Research artifact. Load with the matching MedGemma base model and clinician oversight.",
    }
    torch.save(package, ADAPTER_PTH)
    print("Saved adapter .pth:", ADAPTER_PTH)
    print("Recommended full Hugging Face model directory:", BEST_MODEL_DIR)
    shutil.copy2(ADAPTER_PTH, ARTIFACT_DIR / ADAPTER_PTH.name)

    if SAVE_FULL_MERGED_PTH:
        merged = trainer.model.merge_and_unload()
        full_state = {k: v.detach().cpu() for k, v in merged.state_dict().items()}
        torch.save({
            "artifact_type": "medgemma-4b-it-merged-state-dict",
            "base_model": MODEL_ID,
            "processor": MODEL_ID,
            "best_eval_loss": float(trainer.state.best_metric),
            "state_dict": full_state,
        }, FULL_PTH)
        print("Saved large merged .pth:", FULL_PTH)
        shutil.copy2(FULL_PTH, ARTIFACT_DIR / FULL_PTH.name)

    # GGUF, ONNX, and TFLite are not emitted here: this is a multimodal
    # Gemma 3 PEFT adapter, and generic conversion would not preserve the
    # image processor, chat template, LoRA modules, and generation behavior.
    artifact_manifest = {
        "run_root": str(RUN_ROOT),
        "best_checkpoint": str(best_checkpoint),
        "artifacts": {
            "huggingface_safetensors_adapter": {
                "status": "available",
                "path": str(HF_ARTIFACT_DIR),
                "files": ["adapter_model.safetensors", "adapter_config.json"],
            },
            "pth_adapter_package": {
                "status": "available",
                "path": str(ARTIFACT_DIR / ADAPTER_PTH.name),
            },
            "pth_merged_full_model": {
                "status": "available" if SAVE_FULL_MERGED_PTH else "not_requested",
                "path": str(ARTIFACT_DIR / FULL_PTH.name),
            },
            "gguf": {
                "status": "not_supported_for_this_pipeline",
                "reason": "Multimodal Gemma 3 LoRA adapter conversion is not provided.",
            },
            "onnx": {
                "status": "not_supported_for_this_pipeline",
                "reason": "Full multimodal model export requires a separate validated export graph.",
            },
            "tflite": {
                "status": "not_supported_for_this_pipeline",
                "reason": "The multimodal decoder and image processor are not TFLite-exported here.",
            },
        },
    }
    ARTIFACT_MANIFEST.write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    print("Saved artifact manifest:", ARTIFACT_MANIFEST)
else:
    print("No .pth written because RUN_TRAINING=False.")


# ## 7. Held-out inference and evaluation
# 
# Automatic checks are deliberately limited to structural validity and transparent field-level comparisons. Exact string match or text overlap is **not** clinical correctness. The final evaluation must be blinded clinician scoring on the untouched test set, including missed critical findings, unsupported claims, uncertainty, action appropriateness, and localization quality.

# In[ ]:


try:
    import peft  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "peft is missing from the active Vanda environment. Install it before "
        "running this script; package installation is intentionally not performed "
        "inside the training process."
    ) from exc

def load_inference_model(adapter_path=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    gpu_preflight(torch, require_gpu=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **make_model_kwargs(torch))
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    if adapter_path is not None:
        bundle = torch.load(adapter_path, map_location="cpu", weights_only=False)
        # Prefer PEFT's native directory loader. It restores the exact
        # adapter_config.json and adapter_model.safetensors produced by
        # SFTTrainer, avoiding partial raw state-dict loading.
        checkpoint_dir = bundle.get("best_checkpoint") or str(BEST_MODEL_DIR)
        adapter_config = Path(checkpoint_dir) / "adapter_config.json"
        if adapter_config.is_file():
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_dir),
                is_trainable=False,
            )
            print("Loaded adapter with PEFT from:", checkpoint_dir)
        else:
            # Compatibility fallback for older .pth packages without a saved
            # PEFT directory. This path reports incompatibilities explicitly.
            config = LoraConfig(**bundle["lora_hparams"])
            model = get_peft_model(model, config)
            missing, unexpected = model.load_state_dict(bundle["state_dict"], strict=False)
            print("Legacy adapter load missing keys:", len(missing), "unexpected:", len(unexpected))
    model.eval()
    return model, processor

def generate_examine(model, processor, image_path, context=""):
    import torch
    prompt = TASK_PROMPT + (f" Additional de-identified context for clinician review: {context}" if context else "")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    image = Image.open(image_path).convert("RGB")
    # Render the chat text first, then pass the image exactly once to the
    # processor. Passing images both to apply_chat_template and processor
    # causes Gemma 3 to raise: multiple values for keyword argument 'images'.
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[rendered], images=[[image]], return_tensors="pt", padding=True
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", text, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else None

def automatic_scores(prediction_text, reference_text):
    pred, ref = parse_json_response(prediction_text), parse_json_response(reference_text)
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return {"json_valid": 0, "criticality_match": np.nan, "normality_match": np.nan, "finding_token_recall": np.nan}
    ref_tokens = set(re.findall(r"[a-z0-9]+", str(ref.get("suspected_findings", "")).lower()))
    pred_tokens = set(re.findall(r"[a-z0-9]+", str(pred.get("suspected_findings", "")).lower()))
    recall = len(ref_tokens & pred_tokens) / max(1, len(ref_tokens))
    return {
        "json_valid": 1,
        "criticality_match": int(str(pred.get("criticality", "")).casefold() == str(ref.get("criticality", "")).casefold()),
        "normality_match": int(str(pred.get("normality", "")).casefold() == str(ref.get("normality", "")).casefold()),
        "finding_token_recall": recall,
    }

print("Inference and scoring utilities defined.")


# In[ ]:


EVAL_DIR = RUN_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
if RUN_HELDOUT_EVALUATION:
    if not RUN_TRAINING or not ADAPTER_PTH.exists():
        raise FileNotFoundError("Train/package the adapter first, or set ADAPTER_PTH to a trusted existing artifact.")
    test_rows = manifest[manifest.split.eq("test")].reset_index(drop=True)
    adapted_model, adapted_processor = load_inference_model(ADAPTER_PTH)
    adapted_records = []
    for _, row in test_rows.iterrows():
        generated = generate_examine(adapted_model, adapted_processor, row.image_path)
        adapted_records.append({
            "file_id": row.file_id, "split": row.split, "reference_response": row.response,
            "adapted_response": generated, **automatic_scores(generated, row.response),
        })
    predictions = pd.DataFrame(adapted_records)
    predictions.to_json(EVAL_DIR / "adapted_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    predictions.to_csv(EVAL_DIR / "adapted_predictions.csv", index=False, encoding="utf-8-sig")
    test_metrics = predictions[[
        "json_valid", "criticality_match", "normality_match", "finding_token_recall"
    ]].mean(numeric_only=True)
    print(test_metrics)

    # Matplotlib-only held-out test summary; no TensorBoard dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_metrics.to_csv(EVAL_DIR / "test_metrics.csv", header=["mean_score"])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(test_metrics.index, test_metrics.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("Held-out test metrics")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(PROJECT_ROOT / "training_plots" / "heldout_test_metrics.png", dpi=160)
    plt.close(fig)
    print("Saved held-out test plot:", PROJECT_ROOT / "training_plots" / "heldout_test_metrics.png")
else:
    print("RUN_HELDOUT_EVALUATION=False. Test data remains untouched.")


# In[ ]:


CLINICIAN_REVIEW_FILE = EVAL_DIR / "clinician_scoring.csv"
review_columns = [
    "file_id", "adapted_response", "finding_correct", "critical_finding_missed",
    "unsupported_claim", "uncertainty_appropriate", "criticality_correct",
    "action_appropriate", "localization_correct", "reviewer_code", "reviewer_notes",
]
if not CLINICIAN_REVIEW_FILE.exists():
    review_template = manifest[manifest.split.eq("test")][["file_id"]].copy()
    for col in review_columns:
        if col not in review_template:
            review_template[col] = ""
    review_template[review_columns].to_csv(CLINICIAN_REVIEW_FILE, index=False, encoding="utf-8-sig")
print("Clinician scoring template:", CLINICIAN_REVIEW_FILE)


# ## 8. Safety, limitations, and recommended extensions
# 
# This is a course research prototype. MultiCaRe is a literature-derived, mixed-modality image collection; the emergency filter does not prove that an image was acquired at triage, and case-report labels can be incomplete or selection-biased. A real deployment would require prospective, patient-level data, external validation, calibration, subgroup analysis, prospective workflow testing, privacy/security review, and regulatory/clinical governance.
# 
# Recommended resources to investigate before expanding the model:
# 
# - **Chest X-ray/report data:** [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/), and [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/). Check credentialing, de-identification, licensing, and report-label quality.
# - **Localization/detection:** [VinDr-CXR](https://vindr.ai/datasets/cxr), [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), and [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation). Use dedicated detectors/segmenters when boxes or masks are safety-critical.
# - **Medical VQA/instruction data:** [VQA-RAD](https://osf.io/89kps/), [SLAKE](https://www.med-vqa.com/slake/), and [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA). Align label definitions before mixing datasets.
# - **Model/engineering:** Google's [MedGemma fine-tuning notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb), [MedGemma model card](https://huggingface.co/google/medgemma-4b-it), and [MedSigLIP](https://huggingface.co/google/medsiglip-448) for image-only retrieval/classification baselines.
# 
# Use the model card and each dataset's terms as authoritative. Never expose HF tokens in notebook output. Do not use generated urgency or action text as an autonomous order.

# ## 9. Run order
# 
# 1. Run setup, filtering, and candidate inspection.
# 2. Have qualified reviewers complete and adjudicate the annotation CSV.
# 3. Validate annotations and create article-level splits.
# 4. Materialize the image/response manifest and inspect sample records.
# 5. On a bfloat16-capable Vanda GPU, set `RUN_TRAINING=True`; this runs exactly 10 epochs, evaluates every epoch, and restores the lowest-`eval_loss` checkpoint.
# 6. Package the selected adapter as `medgemma_examine_best_adapter.pth`; optionally export the large merged state dict.
# 7. Set `RUN_HELDOUT_EVALUATION=True` and score predictions on the untouched test split.
# 8. Complete blinded clinician review before drawing conclusions or connecting a chatbot to clinical workflow.
