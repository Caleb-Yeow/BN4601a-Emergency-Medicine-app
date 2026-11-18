# BN4601A: A&E triage assistant

dataset: https://huggingface.co/datasets/OpenMed/multicare-cases

## Layout

One folder per part of the app. Each has its own `notebooks/` (experiments, numbered), `src/` (functions the others
import), `models/` (trained files) and a README. Data and outputs stay out of git.

```
multicare_filter/   data prep: picks MultiCaRe ED cases (image + case text + vitals) for testing and the demo
vitals/             Model 2: vitals -> NEWS2, p_urgent, safety floor, note     (VitalsModel().assess())
reasoning/          Model 3: patient card -> triage, chat, SBAR report          (triage(), chat(), report())
```

Pipeline: image -> Model 1 findings, vitals -> Model 2 note + floor -> patient card -> Model 3 -> `enforce_floor()` -> chat -> report.

## My part (Caleb): `multicare_filter/` and `vitals/`

**`multicare_filter/`: test and demo cases.** Rule-based selection from MultiCaRe of cases that fit an A&E app: open
licence, X-ray / CT / ultrasound (or acute clinical photos), adult, case text says the patient came to an emergency
department, at least one vital sign mentioned, no post-op or follow-up images. Default result: **2,340 images from
1,015 cases**, each with a keyword guess of the department. Runs on Colab (`MultiCaRe_filter_colab.ipynb`) or locally
(`notebooks/01_filter_multicare.ipynb`). Details: [multicare_filter/README.md](multicare_filter/README.md).

**`vitals/`: Model 2.** Takes a patient's vital signs and returns:

- a **NEWS2** score and risk band, with a safety extension for missing vitals
- **`p_urgent`**: chance the patient needs KTAS 1–2 care, from a gradient-boosting classifier trained on Korean ED
  triage data (Kaggle KTAS, 1,267 visits). Test AUROC 0.78 vs 0.69 for NEWS2
- a **safety floor**: the least urgent priority the patient may get
- a one-line **note** that goes into Model 3's prompt

```python
from vitals_model import VitalsModel, enforce_floor
out = VitalsModel().assess(rr=26, spo2=89, sbp=112, hr=124, temp_c=38.6, consciousness="alert", age=71)
out["note"]                                         # into Model 3's prompt
final = enforce_floor(llm_priority, out["floor_priority"])   # after Model 3 answers
```

The floor is applied in code after Model 3, so the language model cannot make a patient less urgent than their vitals
allow. Vitals alone miss many urgent patients (e.g. chest pain with normal vitals), so `p_urgent` is an input to
Model 3, not the triage decision. External validation on US data (NHAMCS 2022) showed NEWS2 and the safety rule
transfer, but `p_urgent` would need retraining on local data. Details, results and data download steps:
[vitals/README.md](vitals/README.md).
