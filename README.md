# BN4601A Emergency Medicine triage app: archive

An archive of the BN4601A group project (NUS, Nov 2026): an emergency department triage assistant that reads messy
nurse notes, vital signs and pictures, then suggests an urgency level, a case file, questions, to-dos and an SBAR handover.

> **Student research prototype. Not a medical device. Do not use it on real patients or enter real patient details.**

This repo holds a snapshot of the group's app, plus Caleb Yeow's own work, test results and tools. The group's working
repo is [likingood16/BN4601A](https://github.com/likingood16/BN4601A); the snapshot is its `main` at commit `502a25b` (16 Nov 2026).

## What is where

| Folder | What it holds |
|---|---|
| [`app_snapshot/`](app_snapshot/) | The full app code: `app/` (server, screens, Gradio, tests), `reasoning/` (MedGemma pipeline), `vision/` (image models' code), `vitals/` (vitals model), `multicare_filter/` |
| [`my_work/`](my_work/) | Caleb's results and local files that are not on the group's `main` |
| [`testing/`](testing/) | Test cases, every test round with its feedback form and screenshots, the real-patient findings, and the test scripts |
| [`patches/chat_fix_16Nov/`](patches/chat_fix_16Nov/) | Optional chat improvement. **Not** applied to `app_snapshot/` |
| [`experiments/image_model_testing/`](experiments/image_model_testing/) | Early image-model pipeline experiments and summary metrics |
| [`docs/`](docs/) | User guide, Gradio guide and the 13 Nov retest sheet (PDF) |
| [`environment/`](environment/) | Exact Python package versions and how to rebuild the environment |

## Run the app

The code is here, but the model weights are too large for git (see below).

```bash
git clone https://github.com/Caleb-Yeow/BN4601a-Emergency-Medicine-app.git
cd BN4601a-Emergency-Medicine-app/app_snapshot/app
pip install -r requirements.txt                      # Python 3.11
python install_models.py <path>/ED_app_full_ai_models.zip
python install_models.py <path>/ED_app_models_update.zip
python tests/test_app.py                             # expected: 124 PASS, 0 FAIL
python server.py                                     # full AI: http://127.0.0.1:8000
python server.py --test                              # MedGemma's writing faked; no MedGemma needed
```

The first full-AI start downloads MedGemma 1.5 4B (`unsloth/medgemma-1.5-4b-it`, about 8 GB). An NVIDIA GPU is recommended.
Checked on 19 Nov 2026 with this repo's code and the two zips below: models installed, `test_app.py` 124 PASS, 0 FAIL.

## Model files (not in this repo)

Keep these two files somewhere safe. The app's full AI cannot run without them.

| File | Size | SHA-256 |
|---|---|---|
| `ED_app_full_ai_models.zip` | 715.3 MB | `048254796041712faf6ed67d196750f22d189d5dcb82e3aaeece9f15ae113794` |
| `ED_app_models_update.zip` | 0.14 MB | `31f9f7d6d15b6ac6bf3ce71acdbe25ef01e7ad85a251a8f10138e1754380008b` |

They were shared in the project group chat, and the group also has a private Hugging Face model repo
(`python install_models.py --hub`, group access needed). They are not published here because they are the group's files and
some were trained on datasets with their own terms.

## Data (not in this repo)

| Data | Get it from |
|---|---|
| KTAS triage data | `kaggle datasets download ilkeryildiz/emergency-service-triage-application --unzip` |
| NHAMCS 2022 | CDC NCHS: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/NHAMCS/stata/` |
| MultiCaRe | Hugging Face `OpenMed/multicare-images` and `OpenMed/multicare-cases` |

Instructions are in `app_snapshot/vitals/README.md` and `app_snapshot/multicare_filter/README.md`.

## Changes from the group's `main`

This repo is public and the KTAS dataset's licence is unclear, so patient-level KTAS content is left out:

- 4 KTAS test notes and their answer-key rows are removed from `app/tests/test_notes/`.
- Outputs are cleared from 5 notebooks that displayed KTAS rows. Their code is unchanged.
- One KTAS record in `app/tests/real_ai_check.py` is replaced with made-up values.
- KTAS rows in the test forms keep only the level and urgent flag; their screenshots are left out.
- NUS HPC IDs, email addresses and local paths are masked, so HPC scripts need your own values.

Rebuilding the group test pack or running `real_ai_check.py` on all 10 notes needs the original KTAS notes from the group repo.

## Test history

| Date | What was tested | Result |
|---|---|---|
| 9 Nov | Gradio app, test mode | 68/68 automatic checks; 40/40 cases matched the key; 15/15 guide steps |
| 10 Nov | Main app, full AI, 40 cases | Form: 33 yes, 5 no; 160 s mean per case; 2 GPU crashes |
| 13 Nov | Retest after group fixes | 40 yes, 7 no of 50 rows; 108 s mean per case; no crashes |
| 14 Nov | Fix check and final QC | Fixes confirmed; found a hidden deterioration warning and a STEMI wording error (10 yes, 2 no, others partial) |
| 16 Nov | 36 of 40 real held-out patients | Text set: 14/20 exact level, 20/20 within one level, 10/10 urgent patients flagged, 3 over-flagged |

Details are in [`testing/README.md`](testing/README.md). The 14 and 16 Nov findings were reported against earlier commits, so
some may already be fixed in this snapshot.

## Credits and licence

Built by the BN4601A Emergency Medicine group. No licence is granted for the group's code: ask the authors before reusing it.
Test pictures keep their CC BY terms; see each `ATTRIBUTION.txt`. MedGemma is subject to Google's terms.
