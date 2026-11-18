#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED-VLM pipeline  -  fine-tune MedGemma-4B-it on a filtered MultiCaRe subset for A&E image triage.
==========================================================================================
Single, self-contained file (the MultiCaRe filter from multicare_filter.py is embedded in SECTION 1).

WHAT IT BUILDS
  image (+ optional clinician note / age / sex)
      stage 1  "findings"   : what is visible (anatomy, abnormality, location, appearance)   <- project task EXAMINE
      stage 2  "assessment" : likely problem(s) + suggested next steps, grounded by RAG
                              (retrieves similar *training-split* cases; leave-one-out during training)
  Output: one .pth file (LoRA adapter + config + RAG corpus + metrics) -> load with load_for_inference().

TOGGLE LIGHTWEIGHT <-> HIGH PERFORMANCE (one switch)
  - edit PROFILE below ("colab" | "hpc" | "auto"), or pass  --profile colab|hpc
  - colab : 4-bit QLoRA, ~1k images, time-budgeted, checkpoints to Google Drive, free T4 (16 GB) safe
  - hpc   : bf16 LoRA (r=32), de-duplicated full dataset, throughput-planned 1-3 epochs inside the walltime

GOOGLE COLAB (T4)
  1) Runtime > Change runtime type > T4 GPU.   2) Add a Colab Secret named HF_TOKEN (and accept the
  MedGemma licence at https://huggingface.co/google/medgemma-4b-it).   3) Upload this file, then:
        %run ed_vlm_pipeline.py --smoke          # ~10-20 min end-to-end check (do this first)
        %run ed_vlm_pipeline.py --profile colab  # real run; if Colab disconnects, re-run with  --resume latest
  Use %run (not !python) so the Drive mount and the download button work.

NUS HPC VANDA
  login node:   python ed_vlm_pipeline.py --prefetch --profile hpc      # compute nodes may have no internet
                python ed_vlm_pipeline.py --write-pbs --profile hpc     # writes ed_vlm_hpc.pbs (edit project id/env)
                qsub ed_vlm_hpc.pbs                                     # resubmit the same script to resume
  Other: --run-name NAME   --resume -1|N|latest   --set key=value (override any profile key, e.g. --set epochs=4)

Research/teaching demo only: outputs are NOT validated clinical advice.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import glob
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import importlib.util
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

# ----------------------------------------------------------------------------------------------
# THE TOGGLE
# ----------------------------------------------------------------------------------------------
PROFILE = "auto"          # "colab" (lightweight)  |  "hpc" (high performance)  |  "auto"
MODEL_ID = "google/medgemma-4b-it"

IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
IN_PBS = "PBS_JOBID" in os.environ


def _cli_value(flag, default=None):
    a = sys.argv
    for i, x in enumerate(a):
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
        if x == flag and i + 1 < len(a):
            return a[i + 1]
    return default


def _resolve_profile_name():
    name = _cli_value("--profile") or os.environ.get("ED_VLM_PROFILE") or PROFILE
    if name == "auto":
        name = "colab" if IN_COLAB else ("hpc" if IN_PBS else "colab")
    if name not in ("colab", "hpc"):
        raise SystemExit(f"Unknown profile {name!r}: use 'colab' or 'hpc'")
    return name


PROFILE_NAME = _resolve_profile_name()
HPC_WORKDIR = os.environ.get("ED_VLM_WORKDIR", os.path.expanduser("~/ed_vlm"))
if PROFILE_NAME == "hpc" and "HF_HOME" not in os.environ:      # keep the 3 GB+ of data off $HOME quota
    os.environ["HF_HOME"] = os.path.join(HPC_WORKDIR, "hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------------------------------
# PROFILES  (every key can be overridden:  --set key=value)
# ----------------------------------------------------------------------------------------------
PROFILES = {
    "colab": dict(
        desc="LIGHTWEIGHT - 4-bit QLoRA, ~1k images, free Colab T4 safe",
        use_fast=True,
        quant="4bit", attn="eager", compute_dtype="auto",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        max_images=1000, max_per_case=2, max_per_article=6,       # max_images caps the TRAIN split only
        max_val_images=150, max_test_images=150,                  # val/test caps, drawn from article-disjoint splits of the full pool
        val_frac=0.10, test_frac=0.10, require_impression=True,
        assess_steps=False,                                       # False: assessment target = impression only (steps labels are noisy)
        eval_no_note=True,                                        # also test the adapter with the clinician note removed
        note_prob=0.7, presentation_chars=240,                    # share of training rows that carry a clinician note
        epochs=2, lr=2e-4, weight_decay=0.0, warmup_ratio=0.05, grad_clip=1.0,   # epochs = CEILING; the plan may use fewer
        per_device_bs=1, eff_bs=16, max_len=1280, workers=2,
        rag_k=2, snippet_chars=350,
        max_val_examples=120, max_test_examples=120,
        n_eval_gen=16, gen_bs=2, max_new_findings=96, max_new_assess=260,
        train_minutes=120, eval_minutes=25,                       # whole run incl. data prep; stays inside free-tier limits
        auto_plan=True, plan_safety=0.88, val_every_steps=40, save_every_steps=20, patience=3,
        merged_export=False, log_every=5,
    ),
    "hpc": dict(
        desc="HIGH PERFORMANCE - bf16 LoRA r=32, de-duplicated full dataset, throughput-planned schedule",
        use_fast=True,
        quant=None, attn="eager", compute_dtype="auto",
        lora_r=32, lora_alpha=64, lora_dropout=0.10,
        max_images=None, max_per_case=2, max_per_article=8,       # images of one case share one target text -> cap them
        max_val_images=400, max_test_images=600,                  # max_images caps TRAIN only; val/test have their own caps
        val_frac=0.05, test_frac=0.05, require_impression=True,
        assess_steps=False, eval_no_note=True,
        note_prob=0.7, presentation_chars=300,
        epochs=3, lr=1e-4, weight_decay=0.01, warmup_ratio=0.05, grad_clip=1.0,  # epochs = CEILING; the plan may use fewer
        per_device_bs="auto", eff_bs=32, max_len=1536, workers=8,
        rag_k=2, snippet_chars=400,
        max_val_examples=400, max_test_examples=600,
        n_eval_gen=160, gen_bs=8, max_new_findings=128, max_new_assess=300,
        train_minutes=19 * 60, eval_minutes=60,                    # 19 h + 1 h eval + export fits a 24 h walltime
        auto_plan=True, plan_safety=0.90, val_every_steps=150, save_every_steps=100, patience=3,
        merged_export=True, log_every=20,
    ),
}
SMOKE = dict(max_images=32, max_val_images=8, max_test_images=8, max_per_case=1, max_per_article=2, epochs=1,
             eff_bs=2, max_val_examples=8,
             max_test_examples=8, n_eval_gen=2, save_every_steps=4, val_every_steps=4, auto_plan=False,
             train_minutes=10, eval_minutes=5, log_every=1, patience=None, merged_export=False,
             max_new_findings=48, max_new_assess=96)
SEED = 42

# ----------------------------------------------------------------------------------------------
# dependencies (pip only if something is missing; set ED_VLM_SKIP_INSTALL=1 to disable)
# ----------------------------------------------------------------------------------------------
def ensure_dependencies():
    need = {"transformers": "transformers>=4.50", "accelerate": "accelerate", "peft": "peft>=0.14",
            "bitsandbytes": "bitsandbytes>=0.45", "huggingface_hub": "huggingface_hub",
            "sklearn": "scikit-learn", "PIL": "pillow", "matplotlib": "matplotlib", "pandas": "pandas",
            "pyarrow": "pyarrow", "numpy": "numpy"}
    missing = [spec for mod, spec in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing missing packages:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


if os.environ.get("ED_VLM_SKIP_INSTALL") != "1":
    ensure_dependencies()

# =============================================================================================
# SECTION 1 - MultiCaRe filter (embedded verbatim from multicare_filter.py; do not edit here,
#             change the CONFIG dict or pass profile 'filter_overrides' instead)
# =============================================================================================

import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

def _dataset_dir(repo_id, patterns):
    """Local folder of a Hugging Face dataset: cached copy if present, else download it."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns, local_files_only=True)
    except Exception:
        print(f"Downloading {repo_id} from Hugging Face (first run only)...")
        return snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns)


def _image_files():
    files = sorted(glob.glob(os.path.join(_dataset_dir("OpenMed/multicare-images", ["data/*.parquet"]),
                                          "data", "*.parquet")))
    if not files:
        raise FileNotFoundError("no image parquet files found for OpenMed/multicare-images")
    return files


def _case_file():
    return os.path.join(_dataset_dir("OpenMed/multicare-cases", ["cases.parquet"]), "cases.parquet")

CONFIG = {
    # Reuse allowed for a non-commercial student project; ND (no derivatives) and unlicensed excluded
    "licenses": ["CC0", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA"],
    # A&E imaging; MRI, PET etc. are rarely first-line in A&E
    "radiology_subtypes": ["x_ray", "ct", "ultrasound"],
    # Clinical photographs kept only when the caption describes an acute A&E problem
    "photo_subtypes": ["skin_photograph", "other_medical_photograph"],
    "include_ecg": False,           # ECGs are A&E-relevant but hard for a VLM; switch on if wanted
    # Keep these as quality checks, not hard clinical eligibility criteria.  A
    # MultiCaRe case can be emergency-relevant without mentioning ED or vitals
    # in the article text.
    "min_caption_chars": 8,
    "exclude_later_timepoints": True,   # drop post-operative / follow-up / healed images
    "exclude_echo": False,          # echocardiograms are many of the ultrasound images; flagged either way
    "min_age": 16,
    "allow_unknown_age": True,
    "require_ed": False,
    "min_vitals": 0,                # set to 1 only for a deliberately strict cohort
}

ACUTE_PHOTO = re.compile(
    r"wound|lacerat|burn|abscess|cellulitis|necroti[sz]|gangren|trauma|injur|bite|fractur|"
    r"dislocat|deformit|ha?ematoma|bruis|ecchymo|swelling|swollen|abrasion|stab|gunshot|"
    r"degloving|amputat|foreign body|impale|crush|blister|erythema|purpura|petechi|rash",
    re.IGNORECASE)
# Photos taken in theatre or of specimens are not what the A&E doctor sees on arrival
NOT_PRESENTATION_PHOTO = re.compile(
    r"intra-?operative|post-?operative|operative (view|field|finding|image)|surgical (field|specimen)|"
    r"specimen|resected|after (surgery|repair|debridement)|during (surgery|the operation)",
    re.IGNORECASE)

# Images from later in the stay (after surgery, at follow-up) are not what triage sees
LATER_TIMEPOINT = re.compile(
    r"post-?operative|follow-?up|after (treatment|therapy|surgery|repair|drainage|antibiotic|operation|"
    r"debridement|intervention|stenting|embolization|resection|discharge)|\d+\s*(months?|weeks?|years?) later|"
    r"\bheal(ed|ing)\b|resolution|resolved|at discharge|on discharge|postoperative day|"
    r"day \d+ (post|after)|\d+(st|nd|rd|th) (postoperative|post-operative) day",
    re.IGNORECASE)
ECHO = re.compile(r"echocardiogra|\bTTE\b|\bTEE\b|doppler", re.IGNORECASE)

ED_WORDS = re.compile(
    r"emergency (department|room|unit|ward|centre|center|service)|accident (and|&) emergency"
    r"|\bA&E\b|casualty department|trauma (bay|centre|center|unit)", re.IGNORECASE)
ED_ABBREV = re.compile(r"\b(to|in|at|from|into) (the |our )?(ED|ER)\b")

DEG = r"\s*(?:°|º|˚|℃|degrees?)\s*"
VITALS = {
    "blood_pressure": r"\b\d{2,3}\s*/\s*\d{2,3}\s*mm\s?hg|blood pressure\D{0,20}\d{2,3}",
    "heart_rate": r"(heart rate|pulse rate|pulse|\bHR\b)\D{0,15}\d{2,3}|\d{2,3}\s*(beats per minute|beats/min|bpm)",
    "resp_rate": r"(respiratory rate|respiration rate|\bRR\b)\D{0,15}\d{1,2}|\d{1,2}\s*breaths",
    "oxygen_sat": r"(oxygen saturation|SpO2|SaO2|O2 sat\w*|saturation)\D{0,25}\d{2,3}\s*%",
    "temperature": rf"\b\d{{2,3}}(\.\d)?{DEG}[CF]\b|\b\d{{2,3}}(\.\d)?\s*℃|temperature\D{{0,15}}\d{{2,3}}",
}
_nocap = lambda p: re.sub(r"\((?!\?)", "(?:", p)    # non-capturing groups: str.contains only needs a match
VITAL_RES = {k: re.compile(_nocap(v), re.IGNORECASE) for k, v in VITALS.items()}

# Rough department label from the case text, for hand-checking (the dataset has no department column).
DEPARTMENTS = {
    "Cardiology": r"cardiolog|myocardial infarction|\bSTEMI\b|NSTEMI|acute coronary|angina|"
                  r"cardiac tamponade|pericardial effusion|aortic dissection|heart failure|arrhythmi|"
                  r"atrial fibrillation|ventricular tachycardia|endocarditis|cardiogenic",
    "Respiratory": r"pneumonia|pulmonary embol|pneumothorax|pleural effusion|empyema|asthma|COPD|"
                   r"respiratory failure|pulmonolog|respiratory medicine|hemoptysis|haemoptysis|"
                   r"bronch|ARDS|lung abscess",
    "Orthopaedics": r"orthop(a)?edic|fracture|dislocation|osteomyelitis|septic arthritis|"
                    r"compartment syndrome|tendon rupture|spinal fracture",
    "Surgery": r"general surg|appendicitis|bowel obstruction|intestinal obstruction|perforat|"
               r"laparotomy|laparoscop|cholecystitis|volvulus|hernia|peritonitis|ischemic bowel|"
               r"ischaemic bowel|pancreatitis|abscess drainage|intussusception",
    "Neurology": r"neurolog|neurosurg|stroke|cerebral infarct|intracranial|subdural|subarachnoid|"
                 r"epidural ha?ematoma|seizure|status epilepticus|meningitis|encephalitis|"
                 r"cerebral venous|hydrocephalus|spinal cord compression",
}
DEPT_RES = {k: re.compile(_nocap(v), re.IGNORECASE) for k, v in DEPARTMENTS.items()}
ACUTE_PHOTO, NOT_PRESENTATION_PHOTO, LATER_TIMEPOINT, ECHO, ED_WORDS, ED_ABBREV = (
    re.compile(_nocap(r.pattern), r.flags)
    for r in (ACUTE_PHOTO, NOT_PRESENTATION_PHOTO, LATER_TIMEPOINT, ECHO, ED_WORDS, ED_ABBREV))


def load_images():
    files = _image_files()
    cols = [c for c in pq.read_schema(files[0]).names if c != "image"]
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)


def load_cases():
    rows = pq.read_table(_case_file()).to_pylist()
    cases = pd.DataFrame([{**c, "article_id": a["article_id"]} for a in rows for c in a["cases"]])
    cases["age"] = pd.to_numeric(cases["age"], errors="coerce")
    return cases


def annotate_cases(cases):
    text = cases.get("case_text", pd.Series("", index=cases.index)).fillna("").astype(str)
    out = cases.copy()
    out["ed_presentation"] = text.str.contains(ED_WORDS) | text.str.contains(ED_ABBREV)
    for k, rx in VITAL_RES.items():
        out[f"v_{k}"] = text.str.contains(rx)
    out["n_vitals"] = out[[f"v_{k}" for k in VITALS]].sum(axis=1)
    counts = pd.DataFrame({d: text.str.count(rx) for d, rx in DEPT_RES.items()})
    best = counts.max(axis=1)
    out["department_guess"] = np.where(best == 0, "Other/unclear", counts.idxmax(axis=1))
    # Two departments tied at the top -> flag for hand-checking
    out["department_tied"] = (counts.eq(best, axis=0).sum(axis=1) > 1) & (best > 0)
    return out


def _column(df, name, default=""):
    """Return a string-safe column even when a dataset revision omits it."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _licence_allowed(values, allowed):
    """Match both short and versioned Creative Commons licence values."""
    values = values.fillna("").astype(str).str.strip().str.upper()
    values = values.str.replace(r"\s+", " ", regex=True)
    allowed = [str(x).strip().upper() for x in allowed]
    return values.isin(allowed) | values.map(
        lambda value: any(value.startswith(item + " ") for item in allowed)
    )


def apply_filters(img, cases, cfg=CONFIG):
    """Return (selected images with case info, funnel table)."""
    df = img.merge(cases, left_on="patient_id", right_on="case_id", how="left")
    caption = _column(df, "caption").fillna("").astype(str)
    image_type = _column(df, "image_type").fillna("").astype(str)
    image_subtype = _column(df, "image_subtype").fillna("").astype(str)
    age = pd.to_numeric(_column(df, "age", np.nan), errors="coerce")

    is_rad = image_type.eq("radiology") & image_subtype.isin(cfg["radiology_subtypes"])
    is_photo = (image_type == "medical_photograph") & image_subtype.isin(cfg["photo_subtypes"]) \
        & caption.str.contains(ACUTE_PHOTO) & ~caption.str.contains(NOT_PRESENTATION_PHOTO)
    is_ecg = image_subtype.eq("ekg") if cfg["include_ecg"] else pd.Series(False, index=df.index)

    steps = [
        ("all images", pd.Series(True, index=df.index)),
        ("licence allows reuse", _licence_allowed(_column(df, "license"), cfg["licenses"])),
        ("A&E modality (X-ray/CT/US, acute photo" + (", ECG" if cfg["include_ecg"] else "") + ")",
         is_rad | is_photo | is_ecg),
        (f"caption >= {cfg['min_caption_chars']} chars", caption.str.len() >= cfg["min_caption_chars"]),
    ]
    if cfg["exclude_later_timepoints"]:
        steps.append(("not a later-stage image (post-op, follow-up, healed)", ~caption.str.contains(LATER_TIMEPOINT)))
    if cfg["exclude_echo"]:
        steps.append(("not an echocardiogram", ~caption.str.contains(ECHO)))
    steps += [
        ("linked to a case", _column(df, "case_id", np.nan).notna()),
        (f"age >= {cfg['min_age']}" + (" or unknown" if cfg.get("allow_unknown_age", False) else ""),
         age.ge(cfg["min_age"]) | (age.isna() & cfg.get("allow_unknown_age", False))),
    ]
    if cfg["require_ed"]:
        steps.append(("presented to emergency department", _column(df, "ed_presentation", False).fillna(False).astype(bool)))
    if cfg["min_vitals"]:
        steps.append((f">= {cfg['min_vitals']} measured vital sign(s) in text",
                      _column(df, "n_vitals", 0).fillna(0) >= cfg["min_vitals"]))

    keep = pd.Series(True, index=df.index)
    funnel = []
    for name, mask in steps:
        keep &= mask.fillna(False).astype(bool)
        sel = df[keep]
        funnel.append({"step": name, "images": len(sel), "cases": sel.case_id.nunique(),
                       "articles": sel.article_id.nunique()})
    out = df[keep].copy()
    out["modality"] = np.select([is_rad[keep], is_photo[keep], is_ecg[keep]],
                                [out["image_subtype"], "acute_photo", "ecg"], "unknown")
    out["is_echo"] = out.caption.fillna("").str.contains(ECHO)
    return out.reset_index(drop=True), pd.DataFrame(funnel)


def load_image_bytes(file_ids):
    """PIL images for a handful of file_ids (for display only)."""
    import io
    from PIL import Image
    out = {}
    for f in _image_files():
        t = pq.read_table(f, columns=["file_id", "image"], filters=[("file_id", "in", list(file_ids))])
        for fid, im in zip(t.column("file_id").to_pylist(), t.column("image").to_pylist()):
            out[fid] = Image.open(io.BytesIO(im["bytes"]))
    return out


# =============================================================================================
# SECTION 2 - PIPELINE
# =============================================================================================
import gc
import hashlib
import torch
from PIL import Image
import matplotlib
if not IN_COLAB and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")                      # headless HPC nodes

PIPE: dict = {}          # resolved profile (filled by main)

SYSTEM_PROMPT = (
    "You are an emergency-department decision-support assistant for clinicians. Describe what is visible, "
    "state uncertainty honestly, and never present an uncertain finding as a definitive diagnosis. "
    "You support, and do not replace, clinical judgement.")
NOT_STATED = "Not specified in source case; senior clinician review advised."
DISCLAIMER = ("AI-generated decision support for a research demo. Not a diagnosis; verify against the "
              "patient and local protocols.")
MODALITY_LABEL = {"x_ray": "X-ray", "ct": "CT", "ultrasound": "ultrasound",
                  "acute_photo": "clinical photograph", "ecg": "ECG"}
LORA_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"   # LLM only, not the vision tower


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


class Budget:
    """Wall-clock guard so a free Colab session / PBS job ends gracefully with a usable model."""
    def __init__(self, minutes):
        self.t0, self.limit = time.time(), (minutes * 60 if minutes else None)

    def expired(self):
        return self.limit is not None and (time.time() - self.t0) > self.limit

    def used_min(self):
        return (time.time() - self.t0) / 60


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------------------------
# environment: Drive / work dirs / HF token / device
# ---------------------------------------------------------------------------------------------
def mount_drive_if_colab():
    if not IN_COLAB:
        return None
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return "/content/drive/MyDrive"
    except Exception as e:                                          # noqa: BLE001
        log("Drive mount failed (%s) - checkpoints stay on the VM and are lost on disconnect!" % e)
        return None


def load_hf_token(required=True):
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok and IN_COLAB:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:                                           # noqa: BLE001
            pass
    if not tok:
        with contextlib.suppress(Exception):
            from huggingface_hub import get_token
            tok = get_token()
    if tok:
        # In PBS offline mode, login() calls whoami() and requires network
        # access even when all model files are already cached.
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = tok
    elif required and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("No HF_TOKEN found. Colab: add a Secret named HF_TOKEN. HPC: export HF_TOKEN=... or "
                           "run `huggingface-cli login`. Also accept the licence at "
                           "https://huggingface.co/google/medgemma-4b-it")
    return tok


def check_model_access(model_id):
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model_id, "config.json")
    except Exception as e:                                          # noqa: BLE001
        raise RuntimeError(f"Cannot access {model_id}. Accept the licence on its Hugging Face page with the same "
                           f"account as your token. Original error: {e}") from e


def setup_device(P):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Colab: Runtime > Change runtime type > T4 GPU. HPC: request ngpus=1.")
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024 ** 3
    dtype = P["compute_dtype"]
    if dtype == "auto":      # native bf16 needs compute capability >= 8 (T4 is 7.5 -> fp16)
        dtype = "bfloat16" if props.major >= 8 else "float16"
    P["_dtype"] = getattr(torch, dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if P["quant"] is None and vram < 24:
        log(f"GPU has only {vram:.0f} GB -> switching to 4-bit QLoRA to avoid OOM")
        P["quant"] = "4bit"
    if P["per_device_bs"] == "auto":
        P["per_device_bs"] = 4 if vram >= 70 else 2 if vram >= 38 else 1
    P["grad_accum"] = max(1, math.ceil(P["eff_bs"] / P["per_device_bs"]))
    log(f"GPU: {props.name} | {vram:.1f} GB | dtype={dtype} | quant={P['quant']} | "
        f"batch={P['per_device_bs']} x accum {P['grad_accum']}")
    return vram


def make_run_dir(P, run_name):
    if IN_COLAB:
        base = (mount_drive_if_colab() or "/content") + "/ed_vlm_runs"
    else:
        base = os.path.join(HPC_WORKDIR if PROFILE_NAME == "hpc" else os.path.expanduser("~/ed_vlm"), "runs")
    run_dir = os.path.join(base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cache = os.path.join("/content" if IN_COLAB else os.path.dirname(base), "img_cache")
    os.makedirs(cache, exist_ok=True)
    return run_dir, cache


# ---------------------------------------------------------------------------------------------
# dataset: filter -> targets -> sample -> split -> image cache
# ---------------------------------------------------------------------------------------------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
DX_RX = re.compile(r"\b(diagnos\w+|consistent with|suggestive of|compatible with|confirmed|revealed|"
                   r"secondary to|attributed to|due to)\b", re.IGNORECASE)
# Rule-based triage ordering of management sentences (NOT learned from outcomes): stabilise -> investigate -> treat -> disposition
STEP_TIERS = [
    ("Immediate stabilisation", re.compile(
        r"\b(resuscitat\w*|intubat\w+|oxygen|ventilat\w+|fluids?|analges\w+|morphine|fentanyl|stabili[sz]\w+|airway|"
        r"transfus\w+|vasopressors?|tourniquet|splint\w*|immobili[sz]\w+)\b", re.IGNORECASE)),
    ("Investigate", re.compile(
        r"\b(blood tests?|blood cultures?|laborator\w+|ECG|electrocardiogra\w+|CT|computed tomography|MRI|ultrasound|"
        r"X-ray|radiograph\w*|angiograph\w+|echocardiogra\w+|biops\w+)\b.{0,60}\b(performed|obtained|ordered|"
        r"requested|done|arranged|was|were)\b", re.IGNORECASE)),
    ("Definitive treatment", re.compile(
        r"\b(antibiotic\w*|anticoagul\w+|thrombolys\w+|drain\w*|surg\w+|underwent|fixation|reduction|embolization|"
        r"repair\w*|resection|debridement|treated|managed|administered|started on|initiated)\b", re.IGNORECASE)),
    ("Disposition", re.compile(
        r"\b(admitted|referred|transferred|observation|monitor\w*|intensive care|ICU|ward)\b", re.IGNORECASE)),
]
LEAK_RX = re.compile(r"\b(diagnos\w+|revealed|showed|demonstrat\w+|confirmed|consistent with|suggestive|compatible|"
                     r"biops\w+|patholog\w+|histolog\w+|CT|MRI|X-ray|radiograph\w*|ultrasound|scan|imaging|"
                     r"echocardiogra\w+)\b", re.IGNORECASE)
# Impression targets: only an explicit diagnosis phrase ("diagnosed with X", "consistent with X", ...), cut at the
# first clause boundary. Sentences about follow-up or past history and negated mentions ("no signs suggestive of X")
# are skipped; cases with no such phrase are dropped by require_impression.
LATER_SENT = re.compile(LATER_TIMEPOINT.pattern + r"|follow[- ]up|\bat \d+ (?:days?|weeks?|months?|years?)\b|\bdischarged\b",
                        re.IGNORECASE)
HISTORY_RX = re.compile(r"\b(?:history of|previously|prior|(?:years?|months?) (?:ago|before|earlier)|(?:in|since) (?:19|20)\d\d|"
                        r"past medical|had been diagnosed|was known)\b", re.IGNORECASE)
FIG_RX = re.compile(r"\s*[\(\[]\s*(?:fig\w*|figure|table|supplementary)[^\)\]]*[\)\]]", re.IGNORECASE)
DX_PHRASE = re.compile(
    r"\b(?:(?:was|were|is|being) (?:subsequently |finally |eventually |then )?diagnosed (?:with|as(?: having)?)"
    r"|(?:final|definitive|definite|working|clinical|presumptive|provisional|histopathological|pathological)? ?diagnosis (?:of|was)"
    r"|consistent with|compatible with|suggestive of|in keeping with|indicative of)\s+(.+)", re.IGNORECASE)
DX_NEG = re.compile(r"\b(?:no|not|without|negative for|ruled out|rule out|excluded|unlikely|absence of|less likely|rather than|"
                    r"instead of)\b[^.]{0,60}$", re.IGNORECASE)
DX_STOP = re.compile(
    r"\s*(?:[.;:,\"]|\(|\[|\s(?:but|or|versus|vs|rather than|located|measuring|involving|was|were|is|are|had|has)\b"
    r"|\sin (?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\s(?:and (?:was|were|he|she|the|underwent|treated|admitted|referred|started))\b"
    r"|\s(?:by|based on|after|following|on the basis of|confirmed by)\b"
    r"|\s(?:who|which|that|for|thus|so|hence|therefore|when|while|since|at|on|during|according"
    r"|in (?:\d|our|the (?:emergency|hospital|clinic|department)))\b)", re.IGNORECASE)
DX_NOT = re.compile(r"^(?:the |a |an )?(?:above|this|these|that|those|it|him|her|them|which|findings?|patient|diagnosis|"
                    r"imaging|ct|mri|x-ray|images?|normal|atypical|unremarkable|previous|prior|our|his|their|same|then|"
                    r"also|both|further|made|rendered|established|confirmed|suspected|considered|what|how|whether|"
                    r"in|on|at|to|with|and|or)\b",
                    re.IGNORECASE)


def _clip(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def clean_caption(c):
    c = " ".join(str(c).split())
    c = re.sub(r"^(fig(ure)?\.?\s*\d+[a-z]?[\.:)]?\s*)", "", c, flags=re.IGNORECASE)
    c = re.sub(r"^\(?\s*[a-h](?:\s*(?:,|and|&|-)\s*[a-h])*\s*\)\s*", "", c, flags=re.IGNORECASE)   # panel labels: (A), a and b)
    return _clip(c, 600)


def _sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(" ".join(str(text).split())) if len(s.strip()) > 25]


def classify_step(s):
    for tier, (_, rx) in enumerate(STEP_TIERS):
        if rx.search(s):
            return tier
    return None


def diagnosis_phrases(sents, n=2):
    """Up to n distinct explicit diagnosis phrases, in narrative order."""
    out = []
    for s in sents:
        for m in DX_PHRASE.finditer(s):
            if DX_NEG.search(s[:m.start()]):
                continue
            ph = DX_STOP.split(m.group(1), maxsplit=1)[0].strip(" ,.-'\"")
            ph = re.sub(r"^(?:a|an|the|of|having)\s+", "", ph, flags=re.IGNORECASE)
            if 3 <= len(ph) <= 120 and len(ph.split()) <= 12 and not DX_NOT.match(ph):
                if ph.lower() not in {p.lower() for p in out}:
                    out.append(ph)
                break
        if len(out) >= n:
            break
    return out


def extract_course(case_text):
    """Weak supervision from the case narrative -> (impression, ordered next steps).
    Follow-up and past-history sentences are dropped. Steps are ordered by clinical-urgency tier, then narrative order."""
    sents = [FIG_RX.sub("", s) for s in _sentences(case_text) if not LATER_SENT.search(s) and not HISTORY_RX.search(s)]
    steps = []
    for s in sents:
        t = classify_step(s) if not DX_PHRASE.search(s) else None
        if t is not None:
            steps.append((t, len(steps), s))
    steps = sorted(steps)[:4]
    lines = [f"{i}. {STEP_TIERS[t][0]}: {_clip(s, 180)}" for i, (t, _, s) in enumerate(steps, 1)]
    return "; ".join(diagnosis_phrases(sents)), "\n".join(lines)


def extract_presentation(case_text, n_chars=300):
    """The 'case description' a clinician would have BEFORE the work-up: opening sentences with no result/diagnosis/management."""
    out = []
    for s in _sentences(case_text)[:6]:
        if LEAK_RX.search(s) or DX_RX.search(s) or classify_step(s) is not None or LATER_TIMEPOINT.search(s):
            continue
        out.append(s)
        if len(out) == 2 or len(" ".join(out)) >= 0.6 * n_chars:
            break
    return _clip(" ".join(out), n_chars)


def balanced_take(df, n):
    """Round-robin over modalities so X-ray/CT/US/photos are all represented."""
    pools = {m: list(g.index) for m, g in df.groupby("modality")}
    picked = []
    while len(picked) < n and any(pools.values()):
        for m in list(pools):
            if pools[m] and len(picked) < n:
                picked.append(pools[m].pop())
    return df.loc[picked]


def split_by_article(df, val_frac, test_frac, seed):
    """Group split: all images/cases of one article stay in one split (no leakage)."""
    grp = df["article_id"].astype(str)
    uniq = np.array(sorted(grp.unique()))
    np.random.RandomState(seed).shuffle(uniq)
    n_test, n_val = max(1, round(len(uniq) * test_frac)), max(1, round(len(uniq) * val_frac))
    test_g, val_g = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    df = df.copy()
    df["split"] = np.where(grp.isin(test_g), "test", np.where(grp.isin(val_g), "val", "train"))
    return df


def split_and_cap(df, P):
    """Article-level split of the WHOLE pool first, then cap each split. Capping before splitting (the old order)
    left only ~25 test images; this keeps val/test article-disjoint from train at any max_images."""
    df = split_by_article(df, P["val_frac"], P["test_frac"], SEED)
    log("Full-pool split sizes (before caps): " + df.split.value_counts().to_dict().__repr__())
    caps = {"train": P["max_images"], "val": P.get("max_val_images"), "test": P.get("max_test_images")}
    parts = []
    for part, cap in caps.items():
        sub = df[df.split == part]
        parts.append(balanced_take(sub, cap) if cap and len(sub) > cap else sub)
    return pd.concat(parts).reset_index(drop=True)


def cache_images(file_ids, out_dir, max_side=896):
    """Stream parquet row-groups once and save only the selected images as JPEG (low RAM, reusable)."""
    need = {f for f in file_ids if not os.path.exists(os.path.join(out_dir, f"{f}.jpg"))}
    if not need:
        return
    log(f"Caching {len(need)} images to {out_dir} ...")
    import pyarrow.parquet as pq_
    for shard in _image_files():
        if not need:
            break
        pf = pq_.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            ids = pf.read_row_group(rg, columns=["file_id"]).column(0).to_pylist()
            if need.isdisjoint(ids):
                continue
            t = pf.read_row_group(rg, columns=["file_id", "image"])
            for fid, im in zip(t.column("file_id").to_pylist(), t.column("image").to_pylist()):
                if fid in need:
                    try:
                        pil = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
                        pil.thumbnail((max_side, max_side))
                        pil.save(os.path.join(out_dir, f"{fid}.jpg"), "JPEG", quality=92)
                    except Exception as e:                           # noqa: BLE001
                        log(f"  skip {fid}: {e}")
                    need.discard(fid)
    if need:
        log(f"  {len(need)} selected images were not found in the parquet shards (dropped).")


def build_dataset(P, run_dir, cache_dir):
    log("Loading MultiCaRe metadata (images + cases) ...")
    img, cases = load_images(), load_cases()
    cfg = {**CONFIG, **P.get("filter_overrides", {})}
    if cfg.get("require_ed") or cfg.get("min_vitals"):
        cases = annotate_cases(cases)                  # regex pass over all case text; only needed for these rules
    sel, funnel = apply_filters(img, cases, cfg)
    funnel.to_csv(os.path.join(run_dir, "filter_funnel.csv"), index=False)
    log("Filter funnel:\n" + funnel.to_string(index=False))
    if sel.empty:
        raise RuntimeError("Filter removed every image - check CONFIG / dataset revision.")

    df = pd.DataFrame({
        "file_id": sel["file_id"].astype(str), "case_id": sel["case_id"].astype(str),
        "article_id": sel["article_id"].astype(str), "modality": sel["modality"].astype(str),
        "caption_clean": _column(sel, "caption").map(clean_caption),
        "age": pd.to_numeric(_column(sel, "age", np.nan), errors="coerce"),
        "gender": _column(sel, "gender").fillna("").astype(str).str.lower(),
        "case_text": _column(sel, "case_text").fillna("").astype(str),
    }).reset_index(drop=True)
    df = df.drop_duplicates("file_id").drop_duplicates(["case_id", "caption_clean"])
    df = df[df.caption_clean.str.len() >= CONFIG["min_caption_chars"]]

    ext = {cid: (*extract_course(t), extract_presentation(t, P["presentation_chars"]))
           for cid, t in df.drop_duplicates("case_id")[["case_id", "case_text"]].itertuples(index=False)}
    df["impression"] = df.case_id.map(lambda c: ext[c][0])
    df["steps"] = df.case_id.map(lambda c: ext[c][1])
    df["note_full"] = df.case_id.map(lambda c: ext[c][2])
    df = df.drop(columns=["case_text"])
    if P["require_impression"]:
        df = df[df.impression.str.len() > 0]
    log(f"{len(df)} images from {df.case_id.nunique()} cases / {df.article_id.nunique()} articles after cleaning")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if P["max_per_case"]:
        df = df[df.groupby("case_id").cumcount() < P["max_per_case"]]
    if P["max_per_article"]:
        df = df[df.groupby("article_id").cumcount() < P["max_per_article"]]
    df = split_and_cap(df.reset_index(drop=True), P)

    cache_images(df.file_id.tolist(), cache_dir)
    df["image_path"] = df.file_id.map(lambda f: os.path.join(cache_dir, f"{f}.jpg"))
    df = df[df.image_path.map(os.path.exists)].reset_index(drop=True)
    # Clinician note: always present for val/test; present for ~note_prob of train rows so the model works with AND without it.
    has_note = df.file_id.map(lambda f: int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % 100 < int(P["note_prob"] * 100))
    df["note"] = np.where((df.split != "train") | has_note, df.note_full, "")
    df[["file_id", "case_id", "article_id", "modality", "split"]].to_csv(os.path.join(run_dir, "splits.csv"), index=False)
    log("Split sizes:\n" + df.groupby(["split", "modality"]).size().unstack(fill_value=0).to_string())
    return df


def show_samples(df, run_dir, n=6):
    import matplotlib
    import matplotlib.pyplot as plt
    s = df[df.split == "train"].sample(min(n, (df.split == "train").sum()), random_state=SEED)
    cols = 3
    fig, axes = plt.subplots(math.ceil(len(s) / cols), cols, figsize=(12, 4.2 * math.ceil(len(s) / cols)))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), s.itertuples()):
        ax.imshow(Image.open(r.image_path))
        ax.set_title(f"{MODALITY_LABEL.get(r.modality, r.modality)} | case {r.case_id}\n{_clip(r.caption_clean, 90)}", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "samples.png"), dpi=120)
    plt.show()


# ---------------------------------------------------------------------------------------------
# RAG over the TRAIN split only (TF-IDF; no extra model, self-contained, rebuilt from the .pth at load time)
# ---------------------------------------------------------------------------------------------
class CaseRetriever:
    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.corpus = corpus
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                                   max_features=300000, dtype=np.float32)
        self.X = self.vec.fit_transform([c["index_text"] for c in corpus])
        self.articles = np.array([c["article_id"] for c in corpus])

    @staticmethod
    def build_corpus(train_df):
        corpus = []
        for cid, g in train_df.groupby("case_id", sort=False):
            r = g.iloc[0]
            caps = _clip(" ".join(g.caption_clean.head(2)), 400)
            corpus.append(dict(case_id=str(cid), article_id=str(r.article_id), modality=r.modality,
                               findings=caps, impression=r.impression, steps=r.steps, presentation=r.note_full,
                               index_text=f"{r.note_full} {caps} {r.impression} {r.steps}"))
        return corpus

    def search(self, queries, k, exclude_articles=None, chunk=256):
        Q, out = self.vec.transform(queries), []
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ self.X.T).toarray()
            if exclude_articles is not None:                       # leave-one-out: never retrieve own article
                ex = np.array(exclude_articles[s:s + chunk])
                S[ex[:, None] == self.articles[None, :]] = -1.0
            kk = min(k, S.shape[1])
            idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            for r in range(S.shape[0]):
                order = idx[r][np.argsort(-S[r, idx[r]])]
                out.append([(int(j), float(S[r, j])) for j in order if S[r, j] > 0])
        return out

    def format_refs(self, hits, chars):
        lines = []
        for n, (j, _) in enumerate(hits, 1):
            c = self.corpus[j]
            if not steps_on():                     # impression-only targets: leave out the noisy management lines
                txt = f"Findings: {c['findings'][:int(.6 * chars)]} | Diagnosis: {c['impression'][:int(.4 * chars)]}"
            else:
                txt = (f"Findings: {c['findings'][:int(.25 * chars)]} | Course: {c['impression'][:int(.30 * chars)]}"
                       f" | Management: {(c['steps'] or 'not stated').replace(chr(10), ' ')[:int(.45 * chars)]}")
            lines.append(f"[{n}] ({MODALITY_LABEL.get(c['modality'], c['modality'])}) {txt}")
        return "\n".join(lines)


def attach_refs(df, retr, P):
    """Pre-compute the retrieved context for every row (queries = modality + findings text)."""
    for part in ("train", "val", "test"):
        m = df.split == part
        if not m.any():
            continue
        sub = df[m]
        queries = (sub.modality + " " + sub.note + " " + sub.caption_clean).tolist()
        hits = retr.search(queries, P["rag_k"], exclude_articles=sub.article_id.tolist() if part == "train" else None)
        df.loc[m, "refs"] = [retr.format_refs(h, P["snippet_chars"]) for h in hits]
    return df


# ---------------------------------------------------------------------------------------------
# prompts / examples
# ---------------------------------------------------------------------------------------------
def patient_line(age, gender):
    bits = []
    if age is not None and not (isinstance(age, float) and math.isnan(age)):
        bits.append(f"{int(age)}-year-old")
    if str(gender).lower() in ("male", "female"):
        bits.append(str(gender).lower())
    return ("Patient: " + " ".join(bits) + ". ") if bits else ""


def _header(r):
    return (f"Image type: {MODALITY_LABEL.get(r.modality, r.modality)}. {patient_line(r.age, r.gender)}"
            f"Clinician note: {r.note or 'none provided'}.")


def prompt_findings(r):
    return (_header(r) + "\nTask: describe the clinically relevant visual findings (anatomy, abnormalities, "
            "location and appearance).")


def steps_on():
    """Profiles saved before assess_steps existed trained with next steps, hence the True default."""
    return bool(PIPE.get("assess_steps", True))


def prompt_assess(r, findings, refs, lit=""):
    lit_block = f"Evidence from published literature (retrieved at consultation time):\n{lit}\n" if lit else ""
    head = (_header(r) + f"\nObserved findings: {findings}\nSimilar past cases from the knowledge base "
            f"(context only, may not apply to this patient):\n{refs or 'none retrieved'}\n{lit_block}")
    if not steps_on():
        return head + "Task: give the most likely diagnosis or problem. Use exactly this format:\nImpression: ..."
    return (head + "Task: give the most likely problem(s) with stated uncertainty, then up to 4 suggested next steps for the "
            "emergency team ordered from most to least suitable (stabilise, investigate, treat, disposition). "
            "Use exactly this format:\nImpression: ...\nSuggested next steps (most to least suitable):\n1. ...\n2. ...")


def target_assess(r):
    if not steps_on():
        return f"Impression: {r.impression}"
    return f"Impression: {r.impression}\nSuggested next steps (most to least suitable):\n{r.steps or '1. ' + NOT_STATED}"


def row_example(r, task):
    if task == "findings":
        return dict(image=r.image_path, prompt=prompt_findings(r), target=r.caption_clean, task=task)
    return dict(image=r.image_path, prompt=prompt_assess(r, r.caption_clean, r.refs), target=target_assess(r), task=task)


def epoch_examples(df, epoch):
    """One example per image per epoch, alternating task; each image sees both tasks across 2 epochs."""
    return [row_example(r, "findings" if (i + epoch) % 2 == 0 else "assess") for i, r in enumerate(df.itertuples())]


def eval_examples(df, cap):
    ex = [row_example(r, t) for r in df.itertuples() for t in ("findings", "assess")]
    return ex[:cap]


def chat_text(processor, prompt, target=None):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    t = processor.apply_chat_template(msgs, add_generation_prompt=target is None, tokenize=False)
    t = t.strip() if target is not None else t
    return t[5:] if t.startswith("<bos>") else t      # the processor/tokenizer adds BOS again


class ExDataset(torch.utils.data.Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class Collator:
    """Tokenises image+chat; labels = assistant answer only (prompt, image tokens and padding masked)."""
    def __init__(self, processor, max_len):
        tok = processor.tokenizer
        self.processor, self.max_len = processor, max_len
        tok.padding_side = "right"
        self.sot = tok.convert_tokens_to_ids("<start_of_turn>")
        self.hdr = len(tok("<start_of_turn>model\n", add_special_tokens=False).input_ids)
        self.warned = False

    def __call__(self, batch):
        texts = [chat_text(self.processor, b["prompt"], b["target"]) for b in batch]
        images = [[Image.open(b["image"]).convert("RGB")] for b in batch]
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        ids, labels = enc["input_ids"], enc["input_ids"].clone()
        for i in range(ids.shape[0]):
            pos = (ids[i] == self.sot).nonzero()
            labels[i, : int(pos[-1]) + self.hdr] = -100
        labels[enc["attention_mask"] == 0] = -100
        if ids.shape[1] > self.max_len and not self.warned:
            log(f"WARNING: sequence of {ids.shape[1]} tokens > max_len {self.max_len}; lower rag_k/snippet_chars if OOM")
            self.warned = True
        out = dict(enc)
        out["labels"] = labels
        return out


def to_device(b, P):
    return {k: (v.to("cuda", dtype=P["_dtype"]) if k == "pixel_values" else v.to("cuda")) for k, v in b.items()}


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
def load_base(P, trainable=True):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    try:
        processor = AutoProcessor.from_pretrained(
            P["model_id"],
            use_fast=bool(P.get("use_fast", True)),
        )
    except (TypeError, ValueError) as exc:
        if not bool(P.get("use_fast", True)):
            raise
        log(f"Fast processor unavailable; falling back to slow processor: {exc}")
        processor = AutoProcessor.from_pretrained(P["model_id"], use_fast=False)
    kw = dict(attn_implementation=P["attn"], device_map={"": 0})   # eager attention is recommended for Gemma 3 training
    if P["quant"] == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=P["_dtype"])
    try:
        model = AutoModelForImageTextToText.from_pretrained(P["model_id"], dtype=P["_dtype"], **kw)
    except TypeError:                                              # older transformers
        model = AutoModelForImageTextToText.from_pretrained(P["model_id"], torch_dtype=P["_dtype"], **kw)
    return model, processor


def attach_lora(model, P):
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with contextlib.suppress(Exception):
        model.enable_input_require_grads()
    cfg = LoraConfig(r=P["lora_r"], lora_alpha=P["lora_alpha"], lora_dropout=P["lora_dropout"], bias="none",
                     target_modules=LORA_REGEX, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def load_adapter_state(model, sd):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, sd)


@torch.no_grad()
def progress_bar(iterable, desc, total=None):
    """Use tqdm when available, with a quiet fallback for minimal installs."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)
    except ImportError:
        return iterable


@torch.no_grad()
def eval_loss(model, loader, P, desc="Validation"):
    model.eval()
    tot, n = 0.0, 0
    for b in progress_bar(loader, desc, total=len(loader)):
        b = to_device(b, P)
        with torch.autocast("cuda", dtype=P["_dtype"]):
            out = model(**b, use_cache=False)
        nt = int((b["labels"][:, 1:] != -100).sum())
        if nt and torch.isfinite(out.loss):
            tot += out.loss.item() * nt
            n += nt
    return tot / n if n else float("nan")


def preflight(model, collate, ex, P):
    """Catch fp16 overflow (Gemma 3 on T4) before spending the budget."""
    loss = eval_loss(model, torch.utils.data.DataLoader(ExDataset(ex[:2]), batch_size=1, collate_fn=collate), P,
                     desc="Pre-flight")
    if not math.isfinite(loss):
        raise RuntimeError("Pre-flight loss is NaN/inf. On a T4 this is the fp16 overflow problem. Try "
                           "--set compute_dtype=\"bfloat16\" (slower, emulated) or --set attn=\"sdpa\"; on HPC use an A100/H100.")
    log(f"Pre-flight OK (loss {loss:.3f})")


# ---------------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------------
def save_ckpt(path, model, opt, sched, scaler, state):
    tmp = path + ".tmp"
    torch.save({"adapter": adapter_state(model), "opt": opt.state_dict() if opt else None,
                "sched": sched.state_dict() if sched else None, "scaler": scaler.state_dict() if scaler else None,
                "state": state}, tmp)
    os.replace(tmp, path)                                          # atomic: a disconnect never leaves a half-written file


def find_resume(run_dir, resume):
    if str(resume) == "-1":
        return None
    if str(resume) == "latest":
        cands = [p for p in glob.glob(os.path.join(run_dir, "*.pt")) if os.path.basename(p) != "best.pt"]
        return max(cands, key=os.path.getmtime) if cands else None
    p = os.path.join(run_dir, f"epoch_{int(resume):02d}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def benchmark_throughput(model, collate, train_df, P, n_warm=2, n_meas=6):
    """Seconds per training example, measured on a mix of both tasks (fwd+bwd, no optimiser step, nothing learned)."""
    bs = P["per_device_bs"]
    n = min(len(train_df), (n_warm + n_meas) * bs)
    sub = train_df.sample(n, random_state=SEED)
    ex = [row_example(r, "findings" if i % 2 == 0 else "assess") for i, r in enumerate(sub.itertuples())]
    loader = torch.utils.data.DataLoader(ExDataset(ex), batch_size=bs, collate_fn=collate, num_workers=min(P["workers"], 4))
    model.train()
    times = []
    for b in loader:
        torch.cuda.synchronize()
        t = time.time()
        b = to_device(b, P)
        with torch.autocast("cuda", dtype=P["_dtype"]):
            loss = model(**b, use_cache=False).loss
        if torch.isfinite(loss):
            loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append((time.time() - t, b["input_ids"].shape[0]))
    meas = times[n_warm:] or times
    return sum(t for t, _ in meas) / sum(k for _, k in meas)


def make_plan(spe, pool, n_val, P, budget):
    """Pick epochs + sample size so the cosine schedule FINISHES inside the time budget (incl. validation passes)."""
    left = max(60.0, budget.limit - (time.time() - budget.t0)) * P["plan_safety"]
    val_cost = n_val * 0.35 * spe                      # forward-only pass costs roughly a third of fwd+bwd(+recompute)
    n = left / spe
    for _ in range(3):
        n_evals = (n / P["eff_bs"]) / P["val_every_steps"] + 2
        n = max((left - n_evals * val_cost) / spe, P["eff_bs"] * 10)
    epochs = int(min(P["epochs"], max(1, n // pool)))
    used = int(min(pool, n // epochs))
    return dict(sec_per_example=round(spe, 3), pool=pool, used=used, epochs=epochs, est_hours=round(used * epochs * spe / 3600, 2))


def train(model, processor, df, P, run_dir, resume, budget):
    from transformers import get_cosine_schedule_with_warmup
    collate = Collator(processor, P["max_len"])
    full_train = df[df.split == "train"].reset_index(drop=True)
    val_ex = eval_examples(df[df.split == "val"], P["max_val_examples"])
    bs, accum = P["per_device_bs"], P["grad_accum"]
    best_path = os.path.join(run_dir, "best.pt")
    state = dict(epoch_idx=0, next_batch=0, global_step=0, best_val=float("inf"), best_step=0, bad_evals=0,
                 history=[], plan=None, ep_loss_sum=0.0, ep_loss_n=0)

    ck = find_resume(run_dir, resume)
    ckpt = None
    if ck:
        log(f"Resuming from {ck}")
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        load_adapter_state(model, ckpt["adapter"])
        state = ckpt["state"]
    preflight(model, collate, eval_examples(df[df.split == "val"], 2), P)

    if state.get("plan") is None:                      # decided ONCE, then frozen into every checkpoint
        if P["auto_plan"]:
            spe = benchmark_throughput(model, collate, full_train, P)
            state["plan"] = make_plan(spe, len(full_train), len(val_ex), P, budget)
        else:
            state["plan"] = dict(sec_per_example=None, pool=len(full_train), used=len(full_train), epochs=P["epochs"])
        log(f"TRAINING PLAN: {state['plan']}")
    plan = state["plan"]
    train_df = full_train if plan["used"] >= len(full_train) else \
        balanced_take(full_train, plan["used"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    steps_per_epoch = math.ceil(math.ceil(len(train_df) / bs) / accum)
    total_steps = steps_per_epoch * plan["epochs"]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=P["lr"], weight_decay=P["weight_decay"], fused=True)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(P["warmup_ratio"] * total_steps)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=P["_dtype"] == torch.float16)
    if ckpt:
        if ckpt["opt"]:
            opt.load_state_dict(ckpt["opt"])
        if ckpt["sched"]:
            sched.load_state_dict(ckpt["sched"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
    val_loader = torch.utils.data.DataLoader(ExDataset(val_ex), batch_size=bs, collate_fn=collate, num_workers=0)
    log(f"Training: {len(train_df)} images x {plan['epochs']} epoch(s) = {total_steps} optimiser steps | "
        f"validating every {P['val_every_steps']} steps on {len(val_ex)} examples | patience {P['patience']}")

    def run_validation(epoch_pos, tr_loss):
        v = eval_loss(model, val_loader, P, desc=f"Validation epoch {epoch_pos:.2f}")
        model.train()
        state["history"].append(dict(step=state["global_step"], epoch=round(epoch_pos, 3), train_loss=tr_loss, val_loss=v))
        log(f"VAL step {state['global_step']} (epoch {epoch_pos:.2f}): train {tr_loss:.4f} | val {v:.4f} | best {state['best_val']:.4f}")
        if math.isfinite(v) and v < state["best_val"] - 1e-4:
            state.update(best_val=v, best_step=state["global_step"], bad_evals=0)
            torch.save({"adapter": adapter_state(model), "val_loss": v, "step": state["global_step"]}, best_path)
            log("  new best model saved")
        else:
            state["bad_evals"] += 1
        return bool(P["patience"]) and state["bad_evals"] >= P["patience"]

    last_path = os.path.join(run_dir, "last.pt")
    stop = by_budget = False
    for epoch in range(state["epoch_idx"], plan["epochs"]):
        examples = epoch_examples(train_df, epoch)
        order = np.random.RandomState(SEED + epoch).permutation(len(examples))
        batches = [order[i:i + bs].tolist() for i in range(0, len(order), bs)]
        start = state["next_batch"]
        loader = torch.utils.data.DataLoader(ExDataset(examples), batch_sampler=batches[start:], collate_fn=collate,
                                             num_workers=P["workers"])
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, run_n, win_sum, win_n, nan_skips = 0.0, 0, 0.0, 0, 0
        ep_sum, ep_n = (state["ep_loss_sum"], state["ep_loss_n"]) if start else (0.0, 0)
        bi = start - 1

        def snapshot():
            state.update(epoch_idx=epoch, next_batch=bi + 1, ep_loss_sum=ep_sum, ep_loss_n=ep_n)
            save_ckpt(last_path, model, opt, sched, scaler, state)

        for bi, b in enumerate(
            progress_bar(loader, f"Training epoch {epoch + 1}/{plan['epochs']}", total=len(batches) - start),
            start=start,
        ):
            b = to_device(b, P)
            with torch.autocast("cuda", dtype=P["_dtype"]):
                loss = model(**b, use_cache=False).loss
            if not torch.isfinite(loss):
                nan_skips += 1
                if nan_skips > 20:
                    raise RuntimeError("Too many non-finite losses (fp16 overflow?). See preflight hint.")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / accum).backward()
            lv = loss.item()
            run_loss += lv; run_n += 1; win_sum += lv; win_n += 1; ep_sum += lv; ep_n += 1

            last_batch = bi == len(batches) - 1
            if (bi + 1) % accum == 0 or last_batch:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, P["grad_clip"])
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                state["global_step"] += 1
                gs = state["global_step"]
                if gs % P["log_every"] == 0:
                    log(f"ep {epoch + 1}/{plan['epochs']} step {gs}/{total_steps} loss {run_loss / max(run_n, 1):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB "
                        f"budget {budget.used_min():.0f}/{P['train_minutes']}min")
                    run_loss, run_n = 0.0, 0
                if gs % P["val_every_steps"] == 0 and not last_batch:
                    stop = run_validation(epoch + (bi + 1) / len(batches), win_sum / max(win_n, 1))
                    win_sum, win_n = 0.0, 0
                    snapshot()
                    if stop:
                        log("Early stopping: validation loss has not improved.")
                        break
                elif gs % P["save_every_steps"] == 0 and not last_batch:
                    snapshot()
                if budget.expired():
                    by_budget = True
                    break

        if by_budget:
            log("Time budget reached: validating and saving. Re-run with --resume latest to continue.")
        if not stop:
            stop = run_validation(epoch + (bi + 1) / max(len(batches), 1), win_sum / max(win_n, 1) if win_n else ep_sum / max(ep_n, 1))
        if by_budget or stop:
            snapshot()
            break
        state.update(epoch_idx=epoch + 1, next_batch=0, ep_loss_sum=0.0, ep_loss_n=0)
        save_ckpt(os.path.join(run_dir, f"epoch_{epoch + 1:02d}.pt"), model, opt, sched, scaler, state)
        shutil.copyfile(os.path.join(run_dir, f"epoch_{epoch + 1:02d}.pt"), last_path)
    return state


# ---------------------------------------------------------------------------------------------
# generation + metrics (lexical proxies; see notes in README of the reply: clinician review still required)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def generate_batch(model, processor, images, prompts, P, max_new_tokens=None):
    tok = processor.tokenizer
    old = tok.padding_side
    tok.padding_side = "left"
    texts = [chat_text(processor, p) for p in prompts]
    enc = processor(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)
    enc = {k: (v.to("cuda", dtype=P["_dtype"]) if k == "pixel_values" else v.to("cuda")) for k, v in enc.items()}
    model.eval()
    with torch.autocast("cuda", dtype=P["_dtype"]):
        out = model.generate(**enc, max_new_tokens=max_new_tokens or P.get("max_new_assess", 300), do_sample=False, use_cache=True)
    tok.padding_side = old
    return [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or in on at to with is are was were be been by for from that this as it its there which "
            "his her their no not".split())


def _toks(s):
    return _WORD.findall(s.lower())


def rouge_l(pred, ref):
    a, b = _toks(pred), _toks(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def unigram_f1(pred, ref):
    a, b = Counter(_toks(pred)), Counter(_toks(ref))
    ov = sum((a & b).values())
    if not ov:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def term_recall(pred, ref):
    terms = {t for t in _toks(ref) if len(t) > 3 and t not in _STOP}
    return len(terms & set(_toks(pred))) / len(terms) if terms else 0.0


def two_stage_predict(model, processor, rows, retr, P, adapter_on, deadline):
    """Real inference path: generate findings -> retrieve with them -> generate impression + next steps."""
    cm = contextlib.nullcontext() if adapter_on else model.disable_adapter()
    fp, ap = [], []
    with cm:
        steps = range(0, len(rows), P["gen_bs"])
        for s in progress_bar(steps, f"{'Adapted' if adapter_on else 'Base'} testing", 
                              total=math.ceil(len(rows) / P["gen_bs"])):
            if deadline.expired():
                break
            chunk = rows[s:s + P["gen_bs"]]
            imgs = [Image.open(r.image_path).convert("RGB") for r in chunk]
            f = generate_batch(model, processor, imgs, [prompt_findings(r) for r in chunk], P, P["max_new_findings"])
            hits = retr.search([f"{r.modality} {r.note} {x}" for r, x in zip(chunk, f)], P["rag_k"])
            refs = [retr.format_refs(h, P["snippet_chars"]) for h in hits]
            a = generate_batch(model, processor, imgs, [prompt_assess(r, x, rf) for r, x, rf in zip(chunk, f, refs)], P, P["max_new_assess"])
            fp += f; ap += a
            log(f"  {'adapted' if adapter_on else 'base   '} generated {len(fp)}/{len(rows)}")
    return fp, ap


METRICS = ("findings_rougeL", "findings_f1", "assess_rougeL", "assess_term_recall", "impression_f1",
           "impression_term_recall", "format_ok", "words_findings", "words_assess")


def impression_of(text):
    m = re.search(r"impression:\s*(.*)", str(text), re.IGNORECASE)
    return m.group(1).strip() if m else ""


def score_example(r, f, a):
    """Per-example lexical proxies (kept per row so report() can bootstrap confidence intervals)."""
    ref_a = target_assess(r)
    pred_imp = impression_of(a)
    fmt = "impression:" in a.lower() and (not steps_on() or "suggested next steps" in a.lower())
    return dict(findings_rougeL=rouge_l(f, r.caption_clean), findings_f1=unigram_f1(f, r.caption_clean),
                assess_rougeL=rouge_l(a, ref_a), assess_term_recall=term_recall(a, ref_a),
                impression_f1=unigram_f1(pred_imp, r.impression), impression_term_recall=term_recall(pred_imp, r.impression),
                format_ok=float(fmt), words_findings=len(f.split()), words_assess=len(a.split()))


def retrieval_only_predict(rows, retr):
    """No-image, no-model baseline: copy the most similar TRAIN case found from modality + clinician note alone.
    If the adapter cannot beat this, it is not using the image."""
    fp, ap = [], []
    for hits in retr.search([f"{r.modality} {r.note}" for r in rows], 1):
        c = retr.corpus[hits[0][0]] if hits else None
        fp.append(c["findings"] if c else "")
        a = f"Impression: {c['impression']}" if c else "Impression:"
        if steps_on() and c:
            a += f"\nSuggested next steps (most to least suitable):\n{c['steps'] or '1. ' + NOT_STATED}"
        ap.append(a)
    return fp, ap


def evaluate(model, processor, df, retr, P, run_dir):
    collate = Collator(processor, P["max_len"])
    test_df = df[df.split == "test"].reset_index(drop=True)
    test_ex = eval_examples(test_df, P["max_test_examples"])
    loader = torch.utils.data.DataLoader(ExDataset(test_ex), batch_size=P["per_device_bs"], collate_fn=collate)
    rows = list(test_df.head(P["n_eval_gen"]).itertuples())
    conds = [("base (no adapter)", False, rows), ("adapted (best)", True, rows)]
    if P.get("eval_no_note"):                     # same images, clinician note blanked: how much comes from the note?
        conds.append(("adapted, no note", True, [r._replace(note="") for r in rows]))
    res, samples = {}, []

    def add(name, rws, fp, ap, m):
        per = [score_example(r, f, a) for r, f, a in zip(rws, fp, ap)]
        m["n_generated"] = len(per)
        if per:
            m.update(pd.DataFrame(per).mean().to_dict())
        res[name] = m
        for r, f, a, s in zip(rws, fp, ap, per):
            samples.append(dict(model=name, file_id=r.file_id, modality=r.modality, note=r.note,
                                ref_findings=r.caption_clean, pred_findings=f, ref_assessment=target_assess(r),
                                pred_assessment=a, **s))

    for name, on, rws in conds:
        log(f"Evaluating {name} on {len(rws)} test images ...")
        m = {}
        if rws is rows:                           # teacher-forced loss only for the with-note conditions
            with (contextlib.nullcontext() if on else model.disable_adapter()):
                tl = eval_loss(model, loader, P, desc=f"Testing {name}")
            m.update(test_loss=tl, test_perplexity=math.exp(min(tl, 20)) if math.isfinite(tl) else float("nan"))
        fp, ap = two_stage_predict(model, processor, rws, retr, P, on, Budget(P["eval_minutes"] / len(conds)))
        add(name, rws, fp, ap, m)
    add("retrieval only (no image)", rows, *retrieval_only_predict(rows, retr), {})
    pd.DataFrame(samples).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    return res


def paired_bootstrap(samples, ref="base (no adapter)", n_boot=10000):
    """95% CIs for every model's mean and its paired difference from the base model (same test images)."""
    rng = np.random.default_rng(SEED)
    base = samples[samples.model == ref].set_index("file_id")
    out = []
    for name, g in samples.groupby("model", sort=False):
        g = g.set_index("file_id")
        common = g.index.intersection(base.index)
        if not len(common):
            continue
        idx = rng.integers(0, len(common), size=(n_boot, len(common)))
        for m in METRICS:
            if m not in g:
                continue
            a = g.loc[common, m].to_numpy(float)
            lo, hi = np.percentile(a[idx].mean(1), [2.5, 97.5])
            row = dict(model=name, metric=m, n=len(common), mean=a.mean(), ci_lo=lo, ci_hi=hi)
            if name != ref:
                d = a - base.loc[common, m].to_numpy(float)
                dlo, dhi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                row.update(delta_vs_base=d.mean(), delta_ci_lo=dlo, delta_ci_hi=dhi, better_in=int((d > 0).sum()))
            out.append(row)
    return pd.DataFrame(out)


def report(res, history, run_dir):
    """Save report-ready metrics tables and plots for this run."""
    import matplotlib.pyplot as plt
    tab = pd.DataFrame(res)
    tab["delta (adapted-base)"] = tab["adapted (best)"] - tab["base (no adapter)"]
    rounded = tab.round(4)
    rounded.to_csv(os.path.join(run_dir, "results.csv"))
    rounded.to_csv(os.path.join(run_dir, "evaluation_metrics.csv"))
    with open(os.path.join(run_dir, "evaluation_metrics.md"), "w", encoding="utf-8") as f:
        f.write(rounded.to_markdown())
    print("\nEvaluation metrics:\n" + rounded.to_string())

    pred_csv = os.path.join(run_dir, "test_predictions.csv")
    if os.path.exists(pred_csv):
        ci = paired_bootstrap(pd.read_csv(pred_csv))
        if not ci.empty:
            ci.round(4).to_csv(os.path.join(run_dir, "evaluation_ci.csv"), index=False)
            print("\nPaired bootstrap 95% CIs (delta = model - base on the same test images):\n"
                  + ci.round(3).to_string(index=False))

    history_df = pd.DataFrame(history or [])
    if not history_df.empty:
        history_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    # One report-ready row for the selected adapted model. The training and
    # validation values come from the validation checkpoint with the lowest
    # validation loss; testing values come from the adapted test evaluation.
    best_row = {}
    if not history_df.empty and "val_loss" in history_df:
        best_history = history_df.loc[history_df["val_loss"].idxmin()]
        best_row.update({
            "best_step": int(best_history.get("step", 0)),
            "best_epoch": float(best_history.get("epoch", float("nan"))),
            "training_loss": float(best_history.get("train_loss", float("nan"))),
            "validation_loss": float(best_history.get("val_loss", float("nan"))),
        })
    if "adapted (best)" in tab.columns:
        best_row.update({
            "testing_loss": float(tab.loc["test_loss", "adapted (best)"]),
            "testing_perplexity": float(tab.loc["test_perplexity", "adapted (best)"]),
        })
        for metric in (*METRICS, "n_generated"):
            if metric in tab.index:
                best_row[metric] = float(tab.loc[metric, "adapted (best)"])
    if best_row:
        best_table = pd.DataFrame([best_row])
        best_table.to_csv(os.path.join(run_dir, "best_model_metrics.csv"), index=False)
        with open(os.path.join(run_dir, "best_model_metrics.md"), "w", encoding="utf-8") as f:
            f.write(best_table.to_markdown(index=False))
        ascii_table = best_table.to_string(index=False)
        with open(os.path.join(run_dir, "best_model_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(ascii_table + "\n")
        print("\nBest-model summary:\n" + ascii_table)

    # Report-ready loss curves. The validation callback runs at least at the
    # end of every completed epoch, so epochs >= 2 appear as separate points.
    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    if not history_df.empty:
        x = history_df["epoch"] if "epoch" in history_df else history_df["step"]
        loss_ax.plot(x, history_df["train_loss"], "o-", label="Training loss")
        loss_ax.plot(x, history_df["val_loss"], "s-", label="Validation loss")
        loss_ax.set_xlabel("Epoch" if "epoch" in history_df else "Optimizer step")
        loss_ax.legend()
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training and validation loss")
    loss_ax.grid(alpha=0.3)
    loss_fig.tight_layout()
    loss_fig.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=160)
    plt.close(loss_fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    keys = ["findings_rougeL", "findings_f1", "assess_rougeL", "impression_f1", "impression_term_recall", "format_ok"]
    keys = [k for k in keys if k in tab.index]
    models = [c for c in tab.columns if not c.startswith("delta")]
    w = 0.8 / len(models)
    x = np.arange(len(keys))
    for i, name in enumerate(models):
        ax[0].bar(x + (i - (len(models) - 1) / 2) * w, tab.loc[keys, name], w, label=name)
    ax[0].set_xticks(x); ax[0].set_xticklabels(keys, rotation=25, ha="right"); ax[0].set_title("Test metrics (higher = better)")
    ax[0].legend()
    if history:
        e = [h["step"] for h in history]
        ax[1].plot(e, [h["train_loss"] for h in history], "o-", label="train")
        ax[1].plot(e, [h["val_loss"] for h in history], "s-", label="val")
        ax[1].set_xlabel("optimiser step"); ax[1].set_title("Loss (train = since last validation)"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "results.png"), dpi=130)
    plt.close(fig)
    log(f"Saved report files under {run_dir}: evaluation_metrics.csv, evaluation_metrics.md, training_history.csv, loss_curves.png, results.png")


# ---------------------------------------------------------------------------------------------
# export / reload  (.pth)
# ---------------------------------------------------------------------------------------------
def export_pth(model, P, corpus, res, state, path):
    import transformers, peft
    sd = {k: v.half() if v.is_floating_point() else v for k, v in adapter_state(model).items()}
    payload = dict(
        format="ed-vlm-lora-v1", base_model_id=P["model_id"], adapter_state_dict=sd,
        lora=dict(r=P["lora_r"], alpha=P["lora_alpha"], dropout=P["lora_dropout"], target_regex=LORA_REGEX),
        system_prompt=SYSTEM_PROMPT, rag_corpus=corpus,
        rag=dict(k=P["rag_k"], snippet_chars=P["snippet_chars"]),
        metrics={k: {m: float(v) for m, v in d.items()} for k, d in (res or {}).items()},
        best_step=state.get("best_step"), best_val_loss=state.get("best_val"), history=state.get("history"),
        profile={k: v for k, v in P.items() if not k.startswith("_")}, seed=SEED,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
        created=datetime.now().isoformat(timespec="seconds"), disclaimer=DISCLAIMER)
    torch.save(payload, path)
    log(f"Saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


def export_huggingface_adapter(model, processor, run_dir):
    """Save the best LoRA adapter in native HF safetensors format."""
    adapter_dir = os.path.join(run_dir, "huggingface_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    log("Saved Hugging Face safetensors adapter:", adapter_dir)
    return adapter_dir


def write_artifact_manifest(run_dir, pth_path, hf_adapter_dir, merged_path=None):
    """Document portable artifacts and conversions not supported by this VLM."""
    manifest = {
        "run_dir": run_dir,
        "artifacts": {
            "huggingface_safetensors_adapter": {
                "status": "available",
                "path": hf_adapter_dir,
                "files": ["adapter_model.safetensors", "adapter_config.json"],
            },
            "pth_adapter_package": {"status": "available", "path": pth_path},
            "pth_merged_full_model": {
                "status": "available" if merged_path else "not_requested",
                "path": merged_path,
            },
            "gguf": {
                "status": "not_supported_for_this_pipeline",
                "reason": "Generic GGUF conversion does not preserve this multimodal VLM adapter and processor.",
            },
            "onnx": {
                "status": "not_supported_for_this_pipeline",
                "reason": "A validated multimodal export graph is required.",
            },
            "tflite": {
                "status": "not_supported_for_this_pipeline",
                "reason": "The full multimodal decoder and image processor are not exported here.",
            },
        },
    }
    path = os.path.join(run_dir, "artifact_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("Saved artifact manifest:", path)


def export_merged_pth(P, adapter_sd, path):
    """Optional: full merged weights (~8.6 GB bf16) as a plain state_dict .pth - HPC only."""
    from peft import LoraConfig, get_peft_model
    model, _ = load_base({**P, "quant": None, "_dtype": torch.bfloat16, "attn": "eager"})
    model = get_peft_model(model, LoraConfig(r=P["lora_r"], lora_alpha=P["lora_alpha"], target_modules=LORA_REGEX,
                                             task_type="CAUSAL_LM"))
    load_adapter_state(model, adapter_sd)
    merged = model.merge_and_unload()
    torch.save({k: v.cpu() for k, v in merged.state_dict().items()}, path)
    log(f"Saved merged weights {path}")


def offer_download(path):
    if not IN_COLAB:
        log("Model file ready:", path)
        return
    from google.colab import files
    try:
        import ipywidgets as w
        from IPython.display import display
        btn, out = w.Button(description=f"Download {os.path.basename(path)}", button_style="success", icon="download"), w.Output()

        def _click(_):
            with out:
                files.download(path)
        btn.on_click(_click)
        display(btn, out)
    except Exception:                                               # noqa: BLE001
        files.download(path)


def fetch_pubmed_evidence(query, n=3, timeout=8):
    """Optional ONLINE evidence at consultation time (NCBI PubMed E-utilities: guidelines/reviews). [] if offline."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = f"({query}) AND (Guideline[pt] OR Practice Guideline[pt] OR Systematic Review[pt] OR Review[pt]) AND emergency"
    try:
        u = base + "esearch.fcgi?" + urllib.parse.urlencode(dict(db="pubmed", term=term, retmax=n, sort="relevance", retmode="json"))
        ids = json.load(urllib.request.urlopen(u, timeout=timeout))["esearchresult"]["idlist"]
        if not ids:
            return []
        u = base + "efetch.fcgi?" + urllib.parse.urlencode(dict(db="pubmed", id=",".join(ids), retmode="xml"))
        root = ET.fromstring(urllib.request.urlopen(u, timeout=timeout).read())
        out = []
        for art in root.findall(".//PubmedArticle"):
            title = "".join(art.find(".//ArticleTitle").itertext())
            abstract = " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText"))
            out.append(dict(pmid=art.findtext(".//PMID"), year=art.findtext(".//PubDate/Year") or "", title=_clip(title, 160),
                            text=_clip(abstract, 450)))
        return out
    except Exception:                                               # noqa: BLE001 - never break a consult because the network failed
        return []


class ConsultEngine:
    """Use inside the demo chatbot:  engine = load_for_inference('model.pth'); engine.consult(PIL_image, note=..., online=True)"""
    def __init__(self, model, processor, retriever, meta):
        self.model, self.processor, self.retr, self.meta = model, processor, retriever, meta
        self.P = dict(meta["profile"])
        PIPE["assess_steps"] = self.P.get("assess_steps", True)      # prompt format must match what the adapter learned
        self.P["_dtype"] = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def consult(self, image, note="", age=None, gender=None, modality="x_ray", online=False):
        P = self.P
        r = SimpleNamespace(modality=modality, age=age if age is not None else float("nan"), gender=gender or "", note=note)
        img = image.convert("RGB")
        findings = generate_batch(self.model, self.processor, [img], [prompt_findings(r)], P, P.get("max_new_findings", 128))[0]
        hits = self.retr.search([f"{modality} {note} {findings}"], self.meta["rag"]["k"])[0]
        refs = self.retr.format_refs(hits, self.meta["rag"]["snippet_chars"])
        n_assess = P.get("max_new_assess", 300)
        assess = generate_batch(self.model, self.processor, [img], [prompt_assess(r, findings, refs)], P, n_assess)[0]
        sources = []
        if online:                                   # draft impression -> PubMed guideline/review search -> regenerate with evidence
            m = re.search(r"impression:\s*(.+)", assess, re.IGNORECASE)
            q = " ".join(_toks(m.group(1))[:12]) if m else ""
            sources = fetch_pubmed_evidence(q) if q else []
            if sources:
                lit = "\n".join(f"[L{i}] PMID {s['pmid']} ({s['year']}): {s['title']}. {s['text']}" for i, s in enumerate(sources, 1))
                assess = generate_batch(self.model, self.processor, [img], [prompt_assess(r, findings, refs, lit)], P, n_assess)[0]
        return dict(findings=findings, assessment=assess, references=[self.retr.corpus[j] for j, _ in hits],
                    literature=sources, disclaimer=self.meta["disclaimer"])


def load_for_inference(pth_path, quant="4bit"):
    """quant: '4bit' (T4/Colab demo) or None (bf16, >=24 GB GPU)."""
    from peft import LoraConfig, get_peft_model
    meta = torch.load(pth_path, map_location="cpu", weights_only=False)
    P = dict(meta["profile"], quant=quant, model_id=meta["base_model_id"])
    setup_device(P)
    model, processor = load_base(P)
    lc = meta["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=0.0, bias="none",
                                             target_modules=lc["target_regex"], task_type="CAUSAL_LM"))
    load_adapter_state(model, {k: v.float() for k, v in meta["adapter_state_dict"].items()})
    model.eval()
    return ConsultEngine(model, processor, CaseRetriever(meta["rag_corpus"]), meta)


# ---------------------------------------------------------------------------------------------
# HPC helpers
# ---------------------------------------------------------------------------------------------
def write_pbs(path, run_name, script):
    txt = f"""#!/bin/bash
#PBS -N ed_vlm_{run_name}
#PBS -q gpu
#PBS -P <YOUR_PROJECT_ID>
#PBS -l select=1:ncpus=18:ngpus=1:mem=120gb
#PBS -l walltime=24:00:00
#PBS -j oe
# One GPU is used. Queue limits: <=2 GPUs, <=48 h; 24 h is the default. Raise walltime AND --set train_minutes=... together.
cd "$PBS_O_WORKDIR"
# --- environment (edit to match how you installed things on Vanda) ---
# module load cuda   (if your env needs it)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate edvlm
export ED_VLM_SKIP_INSTALL=1          # no internet assumed on compute nodes: install packages beforehand
export HF_HUB_OFFLINE=1               # run `python {script} --prefetch --profile hpc` once on the login node first
export HF_HOME="${{ED_VLM_WORKDIR:-$HOME/ed_vlm}}/hf_cache"
export TOKENIZERS_PARALLELISM=false
python {script} --profile hpc --run-name {run_name} --resume latest   # resubmit this same job to continue after a timeout
"""
    with open(path, "w") as f:
        f.write(txt)
    log(f"Wrote {path}  ->  qsub {path}")


def prefetch(P):
    from huggingface_hub import snapshot_download
    load_hf_token()
    check_model_access(P["model_id"])
    log("Downloading model ...")
    snapshot_download(P["model_id"], ignore_patterns=["*.gguf", "*.msgpack", "*.h5"])
    log("Downloading MultiCaRe images + cases ...")
    _image_files(); _case_file()
    log("Prefetch done. You can now run with HF_HUB_OFFLINE=1 on compute nodes.")


# ---------------------------------------------------------------------------------------------
# MAIN  (follows the pseudocode order)
# ---------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="ED-VLM pipeline")
    ap.add_argument("--profile", choices=["colab", "hpc"], default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default="-1", help="-1 = fresh start (default) | N = epoch_N checkpoint | latest")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--write-pbs", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    P = copy.deepcopy(PROFILES[PROFILE_NAME])
    P.update(profile=PROFILE_NAME, model_id=MODEL_ID)
    if args.smoke:
        P.update(SMOKE)
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"Invalid --set value {kv!r}; use --set KEY=VALUE")
        k, v = kv.split("=", 1)
        try:
            P[k] = ast.literal_eval(v)
        except Exception:                                           # noqa: BLE001
            P[k] = v
    PIPE.update(P)
    run_name = args.run_name or ("smoke" if args.smoke else f"{PROFILE_NAME}_{datetime.now():%Y%m%d}")
    log(f"PROFILE = {PROFILE_NAME}: {P['desc']}")

    if args.write_pbs:
        write_pbs("ed_vlm_hpc.pbs", run_name, os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0].endswith(".py") else "ed_vlm_pipeline.py")
        return
    if args.prefetch:
        prefetch(P)
        return

    # 1-2  libraries (top of file), Drive, token, models/datasets straight from the Hub
    load_hf_token()
    check_model_access(P["model_id"])
    set_seed(SEED)
    setup_device(P)
    run_dir, cache_dir = make_run_dir(P, run_name)
    budget = Budget(P["train_minutes"])
    json.dump({k: v for k, v in P.items() if not k.startswith("_")}, open(os.path.join(run_dir, "run_config.json"), "w"), indent=2)
    log("Run dir:", run_dir)

    # 3  filter + split + sample plot
    df = build_dataset(P, run_dir, cache_dir)
    show_samples(df, run_dir)
    corpus = CaseRetriever.build_corpus(df[df.split == "train"])
    retr = CaseRetriever(corpus)
    df = attach_refs(df, retr, P)
    log(f"RAG corpus: {len(corpus)} train cases")

    # 4  model + LoRA, then 5-6 train (resume-able)
    model, processor = load_base(P)
    model = attach_lora(model, P)
    state = train(model, processor, df, P, run_dir, args.resume, budget)

    # 7  reload best, test, report
    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_path):
        load_adapter_state(model, torch.load(best_path, map_location="cpu", weights_only=False)["adapter"])
        log(f"Loaded best model (step {state['best_step']}, val_loss {state['best_val']:.4f})")
    res = evaluate(model, processor, df, retr, P, run_dir)
    report(res, state["history"], run_dir)

    # 8  .pth export (+ download button on Colab)
    pth = os.path.join(run_dir, f"ed_vlm_{PROFILE_NAME}.pth")
    export_pth(model, P, corpus, res, state, pth)
    hf_adapter_dir = export_huggingface_adapter(model, processor, run_dir)
    merged_path = None
    if P["merged_export"]:
        asd = adapter_state(model)
        del model
        gc.collect(); torch.cuda.empty_cache()
        merged_path = os.path.join(run_dir, f"ed_vlm_{PROFILE_NAME}_merged.pth")
        export_merged_pth(P, asd, merged_path)
    write_artifact_manifest(run_dir, pth, hf_adapter_dir, merged_path)
    offer_download(pth)
    log("DONE")


if __name__ == "__main__":
    main()
