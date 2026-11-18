"""Write the final QC (14 Nov, main 437ca2e) feedback form and zip it with the screenshots.
Usage: python retest4_form.py <retest4 dir> <Gradio_app>"""
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

S, G = Path(sys.argv[1]), Path(sys.argv[2])
s = {x["step"]: x for x in json.load(open(S / "steps.json", encoding="utf8"))}


def one(t):
    return " ".join(t.split())


HOW = " (sent by a script with the same request the button sends; full AI, main 437ca2e, 14 Nov)"
cols = ["your_name", "app (main / Gradio)", "case_or_step", "what_you_did", "what_the_guide_says", "what_happened",
        "same_as_guide (yes/no)", "how_bad (1 cosmetic - 5 unsafe)", "screenshot_file", "time_taken_s", "comments"]
a, ad, c, sb = s["T01 answer consciousness"], s["T01 add trop + ECG"], s["T01 chat"], s["T01 Write SBAR handover"]
pr, wn, fx = s["T01 Change priority (doctor)"], s["T01 worse note after the change"], s["T01 Fix numbers with the worse vitals"]
p5, c5, p11, g3 = s["P05 submit"], s["P05 chat"], s["P11 picture rule"], s["Three groups"]

rows = [
    ("T01", "Inserted text from cases/T01.txt, Submit" + HOW, "Flagged URGENT; chest-pain to-dos",
     f"Level 2, group Urgent; question: {' | '.join(s['T01 submit']['questions'])}; "
     f"to-dos: {' | '.join(s['T01 submit']['todos'])}",
     "yes", "", "screenshots/T01_1_submitted.png", s["T01 submit"]["secs"],
     "Laptop, NVIDIA RTX 4050 6 GB GPU, Windows 11, full AI run locally (main app). "
     "app/tests/test_app.py: 118 PASS, 0 FAIL"),
    ("Fix 3: 'Consciousness not recorded' goes once a later note says alert (T01)",
     "Answered 'What is the level of consciousness (AVPU/GCS)?' with 'alert, GCS 15', Add" + HOW,
     "HANDOVER 4: 'X not recorded' is removed once a later note gives X; ACVPU tile updates",
     f"ACVPU tile: not recorded -> A; case-file lines: {a['report_lines']}; "
     f"timeline: {[e['text'] for e in a['timeline']]}",
     "yes", "", "screenshots/T01_2_after_GCS_answer.png", a["secs"], "FIXED"),
    ("T01: Add trop + ECG", "Typed 'trop 85, ECG ST depression V4-6', Add" + HOW,
     "Added to the case file (Investigations)", "In Investigations: 'trop 85, ECG ST depression V4-6'; level stays 2",
     "yes", "", "", ad["secs"], ""),
    ("Fix 1: chat doesn't ask for tests already done (T01)", f"Chat: '{c['question']}'" + HOW,
     "HANDOVER 4: no 'obtain ECG and troponin' when they're in the notes; leftovers marked '(already done ...)'",
     '"' + one(c["answer"]) + '"', "yes", "3", "screenshots/T01_3_chat.png", c["secs"],
     "FIXED for the repeat request (no ECG/troponin asked again). NEW clinical error: calls it 'ST-elevation "
     "myocardial infarction (STEMI)' and says 'activate the cardiac catheterization lab for emergent PCI', but the "
     "note says ST DEPRESSION + trop 85 (that points to NSTEMI, not STEMI)"),
    ("Fix 1 + 2: Write SBAR handover (T01)", "SBAR handover tab, Write SBAR handover, then Download" + HOW,
     "HANDOVER 4: no 'Okay, here is' opening; no request for tests already done",
     "Opens with '**SBAR Handover**' (no 'Okay, here is'); S/B/A/R all there with the real vitals, GCS 15, ST "
     "depression V4-6 and history; later notes and the chat question copied word for word under it; Download works "
     f"(HTTP {sb['download_http']})",
     "partly", "3", "screenshots/T01_4_SBAR.png", sb["secs"],
     "Opening FIXED. Repeat request mostly fixed, but the Recommendation still says 'consider obtaining a full "
     "cardiac workup including cardiac enzymes' (= troponin, already done) with no '(already done)' mark, and the "
     "AI-written part leaves out troponin 85 (it is only in the copied notes). Same STEMI error as the chat: 'likely "
     "STEMI ... activating the cardiac catheterization lab for emergent PCI' with ST depression. Also an off-topic "
     "qSOFA/sepsis sentence in Assessment"),
    ("New: doctor's Change priority (T01)",
     "As Dr Lim (Doctor): level 3, reason 'pain settled, reviewed at bedside'; then as Triage nurse tried level 4" + HOW,
     "HANDOVER 4: a doctor can lower or raise the level; the AI's level stays underneath; only doctors",
     "Board: level 3, group 'Can wait', 'Level 3, set by Dr Lim: pain settled, reviewed at bedside (AI + safety "
     f"rules said 2)'; timeline: '{pr['timeline'][0]['text']}'; nurse refused: {pr['nurse_refused'][0]} "
     f"'{pr['nurse_refused'][1]}'",
     "yes", "", "screenshots/T01_5_priority_changed.png screenshots/board_1_after_priority.png", "", "Works"),
    ("New: worse note after the doctor lowered the priority (T01)",
     "After Dr Lim's level 3: Add 'now drowsy, responds to voice only, BP 78/50, HR 128'" + HOW,
     "HANDOVER 4: red 'New information since Dr Lim's change ... please review' if new information makes the AI "
     "more worried",
     "NO warning. Board stays level 3 'Can wait'; tiles still HR 112, BP 90/60, ACVPU A; the new numbers only appear "
     "as a text line in the case file's Observations. The app did make 'please check' warnings (HR 128, BP 78/50, "
     "consciousness verbal), but the case page hides every warning that starts with 'please check' "
     "(app/frontend/src/Case.jsx line 66), and found.check is empty, so no tile says 'check'",
     "no", "4", "screenshots/T01_6_review_warning.png screenshots/board_2_review_warning.png", wn["secs"],
     "A deteriorating patient that a doctor has down-graded shows NO visible sign of the new vitals and no review "
     "warning unless someone presses Fix numbers. Suggest: show the 'please check' lines (or mark the tiles 'check'), "
     "and/or let worse later vitals count for the review rule"),
    ("Workaround: Fix numbers with the worse vitals (T01)",
     "Fix numbers: HR 128, BP 78/50, consciousness verbal, Save and re-score" + HOW,
     "Review warning appears when the AI is more worried",
     f"Re-scored in {fx['secs']} s: NEWS2 8 -> {fx['news2']}, urgency 92% -> {fx['ai_p']:.0%}; board keeps Dr Lim's "
     f"level 3 with '{fx['review']}'; case file: {fx['corrected']}",
     "yes", "", "screenshots/T01_7_after_fix_review.png screenshots/board_4_after_fix_review.png", fx["secs"],
     "The review warning works once the numbers are in the tiles"),
    ("Extra: false 'please check' readings from the worse note (T01)", "Same note as above",
     "Only real numbers are read",
     "Warnings made from the notes: \"please check Age: a later note says 85 ('trop 85')\" (troponin read as the age) "
     "and \"please check Sex: a later note says female ('now drowsy')\"",
     "no", "2", "", "", "Hidden on the case page today (see above), but they would show once the 'please check' "
                        "lines are made visible"),
    ("P05", "Inserted text from cases/P05.txt, added picture pictures/child_wrist_normal.png, Submit" + HOW,
     "Child cut-off; record whether anything is marked; whiteboard row and pain line",
     f"Amber 'needs vitals: {' and '.join(p5['missing'])} not recorded', level {p5['level']}, group {p5['group']}; "
     f"pain line {p5['pain_lines']}; whiteboard row: F, 7, 'Fall from scooter, L wrist sore, moving it', 'Needs "
     f"vitals: BP and RR not recorded', Urgent; nothing marked on the X-ray; to-dos: {' | '.join(p5['todos'])}",
     "yes", "1", "screenshots/P05_2_case_file.png screenshots/board_3_three_groups.png", p5["secs"],
     "Pain line still fixed. Still 'Trauma: primary survey' for a child moving a sore wrist (reported before)"),
    ("P05 chat", "Chat: 'Why is this child flagged urgent?'" + HOW, "NEWS2 low; urgent only because vitals are missing",
     '"' + one(c5["answer"]) + '"', "yes", "1", "screenshots/P05_1_chat.png", c5["secs"],
     "Correct. Still lists temperature and consciousness as missing signs (the label says BP and RR)"),
    ("New: picture rule (P11)",
     "Inserted text from cases/P11.txt, added picture pictures/ct_head_bleed_epidural.jpg, Submit" + HOW,
     "HANDOVER 4: a red head-CT bleed moves the patient up (urgent, level at most 2) with the reason shown",
     f"Level {p11['level']}, flagged, group Urgent; reason: '{p11['reason']}'; marks: {p11['marks']}",
     "yes", "", "screenshots/P11_1_picture_rule.png screenshots/P11_2_files.png", p11["secs"],
     "Works; shown on the case page and the whiteboard"),
    ("New: three groups on the whiteboard", "Added T04 and T07, looked at the whiteboard" + HOW,
     "HANDOVER 4: Urgent (flagged or level 1-2), Can wait (level 3), Minor (level 4-5)",
     "; ".join(f"{r['bed']} {r['who']} level {r['level']} -> {r['group']}" for r in g3["rows"]),
     "yes", "", "screenshots/board_3_three_groups.png", "",
     "Correct for every patient, including T01 under Can wait after the doctor's level 3"),
]

out = G / "ED_app_final_QC_14Nov"
(out / "screenshots").mkdir(parents=True, exist_ok=True)
with open(out / "feedback_form_final_QC_14Nov.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for r in rows:
        w.writerow(["Caleb", "main (NVIDIA GPU 6GB)", *r])
for f in (S / "screenshots").glob("*.png"):
    shutil.copy(f, out / "screenshots" / f.name)
z = G / "ED_app_final_QC_14Nov_Caleb.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
    for f in sorted(out.rglob("*")):
        if f.is_file():
            zz.write(f, f.relative_to(out))
miss = [p for r in rows for p in r[6].split() if not (out / p).exists()]
print(len(rows), "rows; screenshots", len(list((out / "screenshots").glob("*.png"))), "; missing:", miss or "none")
print(z, f"{z.stat().st_size / 1e6:.1f} MB")
