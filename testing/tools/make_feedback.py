"""Build the filled feedback form from the walkthrough results + extra findings.
Usage: python make_feedback.py <walkthrough_results.csv> <out csv>"""
import csv
import sys

walk = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
cols = ["your_name", "app (main / Gradio)", "case_or_step", "what_you_did", "what_the_guide_says", "what_happened",
        "same_as_guide (yes/no)", "how_bad (1 cosmetic - 5 unsafe)", "screenshot_file", "time_taken_s", "comments"]
HOW = "Scripted: called the same functions the Gradio buttons call, test mode, 9 Nov; not clicked in a browser. "
rows = []
for r in walk:
    rows.append({**{c: "" for c in cols}, **r, "your_name": "Caleb", "app (main / Gradio)": "Gradio",
                 "comments": (HOW + r.get("comments", "")).strip()})

extra = [
    ("Other cases T03-R10", "Ran all 40 test-pack cases through the Gradio app",
     "Same as expected_results.csv", "40 of 40 identical (level, flag, numbers, to-dos, questions, picture type and "
     "findings); no errors; 45 s for all 40", "yes", "", "Automatic button test test_app.py: 68 of 68 PASS"),
    ("Extra: HR 1500 (typo)", "Note with 'HR 1500' (my case C04)",
     "Like T08: impossible HR ignored WITH a red warning",
     "HR tile says 'not recorded' but NO warning appears (T08's HR 400 does get one). 4-digit values seem not to be "
     "read at all, so the nurse isn't told the typed value was dropped", "no", "3",
     "Suggest: warn on any HR/BP/RR number that is read but out of range, including 4 digits"),
    ("Extra: unresponsive, no vitals", "'found collapsed, unresponsive, no obs possible yet' (my case C03)",
     "Should be top priority",
     "Flagged urgent, 95%, raised to level 2 by the urgency score and vitals floor; never level 1", "no", "3",
     "An unresponsive patient is level 1 in KTAS/ESI. The vitals floor (my part) stops at 2; I can make "
     "'unresponsive or vitals unmeasurable' set floor 1. No test-pack case reaches level 1"),
    ("Extra: mechanism of injury", "'fell off e-scooter, hit head' (C06); P01 'fell off ladder'",
     "Mechanism recognised", "To-do still says 'Trauma: mechanism of injury not recorded' although the note gives it",
     "no", "2", "Same in P01 and in expected_results.csv"),
    ("Extra: normal adult X-ray", "C05: twisted ankle + pictures/adult_leg_no_fracture.jpg",
     "Guide P03: about 1 in 4 normal adult X-rays get a mark", "RED Fracture marked on the normal X-ray", "yes", "3",
     "Known limit (P03), but a red mark on a normal X-ray could push a doctor towards a fracture: worth saying "
     "'possible' rather than red in the demo"),
    ("Extra: troponin added", "C02 chest pain, normal vitals; then Add 'trop 85 ng/L, ECG ST depression V4-6'",
     "(not in guide) a positive troponin should matter",
     "Level stays 2, urgency 22% unchanged, no new to-do", "no", "2",
     "Test mode only (to-dos come from fixed lists); please check with the full AI"),
    ("Extra: two tabs edit the same row", "two users set the doctor on the same patient at the same time",
     "Board stays consistent", "No error; last change wins with no warning to the other user", "yes", "2",
     "Fine for a prototype; maybe show 'changed by X' on the board"),
    ("Extra: chat placeholder", "Chat in test mode", "Placeholder answer",
     "Placeholder starts by echoing an internal line '(Time now: ... Arrived: ...)'", "yes", "1", "Cosmetic, test mode only"),
]
for step, did, guide, happened, same, bad, note in extra:
    rows.append({**{c: "" for c in cols}, "your_name": "Caleb", "app (main / Gradio)": "Gradio", "case_or_step": step,
                 "what_you_did": did, "what_the_guide_says": guide, "what_happened": happened,
                 "same_as_guide (yes/no)": same, "how_bad (1 cosmetic - 5 unsafe)": bad,
                 "comments": (HOW if step.startswith("Extra") else "") + note})

with open(sys.argv[2], "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows written")
