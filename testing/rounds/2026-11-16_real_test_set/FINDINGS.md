# Real held-out test set (16 Nov 2026)

App: group `main` 74287be, full AI, laptop RTX 4050 6 GB. Each note was sent the way Submit sends it. I reviewed each output
against the note, the true level and the paper's picture caption. This review was done by Claude for Caleb. **It is not a clinician's
review.** Run paused at the user's request: A01-A20 and B01-B16 finished, **B17-B20 were not run**.

Per-case results are in `results.csv`. Notes are not repeated there. KTAS notes and vitals are not published here.

## Set A: text + vitals (20 patients with a true triage level)

| Measure | Result |
|---|---|
| Exact level | 14 of 20 |
| Within one level | 20 of 20 |
| Truly urgent patients flagged | 10 of 10 |
| Non-urgent patients flagged urgent | 3 (A01, A04, A13) |
| One level too low | A06, A14 (neither was truly urgent) |
| One level too high | A01, A04, A07, A13 |
| Mean time per case | 81 s |

## Set B: text + vitals + picture (16 of 20 run, true level unknown)

Mean time per case: 265 s. Levels were judged for reasonableness only.

## Problems found, most serious first

1. **Picture readings can be invented.** All ECG readings looked like one template and contradicted the paper in B04 and B07. B13 (a chart) got
   an invented ECG reading. B05's CT reading made up a surgical history that the note contradicts.
2. **Possible under-triage.** B02 has peritonitis signs and abnormal vitals but got level 3. B04 (shock, heart block, STEMI) got level 2, arguably 1.
3. **The number checker discards correct AI values** because of a too-strict word match. B08 showed baseline HR 50 instead of the ER rate of
   30-40; B12 and B15 lost sex (and B15 age); B09 lost ambulance arrival.
4. **Negation is missed.** "Without chest pain" (B06) and "without any notion of trauma" (B16) still produced those pathways.
5. **Off-topic pathways and missing actions.** Seizure (A04), chest pain (A01, A07) and sepsis (A06) to-dos without support; no bleeding to-do for
   melena (A15); no oxygen to-do at SpO2 70% (B12); no ECG with marked tachycardia (A20).
6. **Head-CT bleed check ran on a chest CT** (B14), and the paper's aortic thrombus was missed.
7. **Formatting.** Triage text in Observations (A15, A17, A18, A20), blank "Not recorded" lines (A12), duplicate questions (A01, A16), and a
   missing complaint turned into "Pain score 0/10" (A11).

## Answer-key notes

B10 is a clinical photo, not an X-ray, and B13 is a chart, not an ultrasound (from the published captions).
