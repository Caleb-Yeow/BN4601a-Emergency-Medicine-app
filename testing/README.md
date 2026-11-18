# Testing archive

All runs used the app on a laptop with an NVIDIA RTX 4050 6 GB GPU. Most cases were sent by script with the same request as the
screen's button, so these runs do not prove every click in a browser.

| Folder | Contents |
|---|---|
| `test_pack/` | The group's T and P cases, CC BY pictures, expected results, blank form. T and P cases are written for testing. KTAS cases R05, R06, R09, R10 are left out |
| `real_test_set_40_no_ktas/` | The real held-out set without its 10 KTAS patients: 10 NHAMCS notes and 20 MultiCaRe case reports with pictures |
| `rounds/2026-11-09_gradio_test_mode/` | Six extra made-up cases (C01-C06) and the Gradio feedback form |
| `rounds/2026-11-10_full_ai_first_run/` | First full-AI form. T01-T05 were entered by hand, the rest by script |
| `rounds/2026-11-13_full_ai_retest/` | 40-case retest, workflow steps, timings and screenshots |
| `rounds/2026-11-14_section4_retest/` | Check of four fixes and the new SBAR handover |
| `rounds/2026-11-14_final_qc/` | Final QC: picture rule, Change priority, three whiteboard groups |
| `rounds/2026-11-16_real_test_set/` | `results.csv` and `FINDINGS.md`. B17-B20 were not run |
| `tools/` | The scripts behind these runs. They contain paths from the original laptop, so treat them as a record and adapt before reuse |

The real-patient review in `FINDINGS.md` was written by Claude and is not a clinician's review.
