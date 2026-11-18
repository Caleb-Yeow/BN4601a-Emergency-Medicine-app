"""Per-case comments for the 13 Nov feedback form: the re-checks of what was reported on 10 Nov (retest sheet,
section 4), worked out from the saved results, plus the workflow rows. Filled in after reading the results."""
import json

BAD = {}                      # case -> how bad (1 cosmetic - 5 unsafe), where there is a problem
NOTE = {}                     # case -> extra comment written by hand after reading the result
KEY_MISMATCH_IS_NO = set()    # cases where a different level / flag from expected_results.csv counts as "no"


def todos(p):
    return [t["text"] for t in p["todos"]]


def has(p, *words):
    return [t for t in todos(p) if any(w in t.lower() for w in words)]


def rechecks(pat):
    """case -> text for the comments column: the 10 Nov issue, what the sheet says should show now, what showed."""
    out = {}

    def add(cid, text):
        if cid in pat:
            out.setdefault(cid, []).append(text)

    for cid, bad_words, label in (("T01", ("asthma", "collapse", "stroke", "poison"), "only the chest-pain pathway"),
                                  ("P11", ("poison", "collapse"), "no poisoning or collapse to-do"),
                                  ("T05", ("poison", "collapse"), "no poisoning or collapse to-do")):
        if cid in pat:
            off = has(pat[cid], *bad_words)
            add(cid, f"Re-check (off-topic pathways, should be {label}): "
                     + ("FIXED, none shown" if not off else "NOT fixed, still shows: " + " | ".join(off)))
    for cid in ("T09", "P05"):
        if cid in pat:
            d = pat[cid]["decision"] or {}
            q = (pat[cid]["questions"] or [""])[0]
            asks = "observations" in q.lower() or "blood pressure" in q.lower()
            add(cid, "Re-check (urgent with no reason when observations are missing): "
                     + (f"FIXED, amber 'needs vitals: {' and '.join(d.get('missing', []))} not recorded'"
                        if d.get("needs_obs") else "NOT fixed, no 'needs vitals' label")
                     + ("; first question asks for the observations" if asks else f"; first question is '{q}' (does NOT ask for the observations)"))
    for cid in ("P04", "P05", "P10"):
        if cid in pat:
            p = pat[cid]
            add(cid, "Re-check (children scored with adult models): "
                     + ("FIXED, label says children's urgency score, NEWS2 and level from adult models"
                        if p.get("child") and p.get("child_score") else "NOT fixed, no children's score label")
                     + (f"; level {p['level']}, {'urgent' if (p['decision'] or {}).get('flagged') else 'not urgent'}"
                        + (" (sheet says level 4)" if cid == "P10" else "")))
    if "T04" in pat:
        extra = has(pat["T04"], "primary survey", "x-ray", "xray")
        add("T04", "Re-check (finger cut got primary survey + X-ray to-dos): "
                   + ("FIXED, to-dos are only: " + " | ".join(todos(pat["T04"])) if not extra
                      else "NOT fixed, still shows: " + " | ".join(extra)))
    if "T05" in pat:
        line = next((l.strip() for l in pat["T05"]["report"].splitlines() if "before arrival" in l.lower()), "")
        add("T05", "Re-check (ambulance numbers looked like arrival numbers): "
                   + (f"FIXED, case file says '{line[:110]}'" if line else "NOT fixed, no 'Before arrival (ambulance crew)' line in the case file"))
    for cid in ("P11", "P12"):
        if cid in pat:
            x = has(pat[cid], "x-ray", "xray")
            add(cid, "Re-check (X-ray to-do although a CT was attached): "
                     + ("FIXED, no X-ray to-do" if not x else "NOT fixed, still shows: " + " | ".join(x)))
    return {k: "; ".join(v) for k, v in out.items()}


def extra_rows(out_dir, res, pat):
    """Rows after R10: the workflow steps, each timed (retest sheet, section 2.4)."""
    f = out_dir / "workflow.json"
    if not f.exists():
        return []
    steps = json.loads(f.read_text(encoding="utf8"))
    rows = []
    how = " (sent by a script with the same request the button sends)"
    for s in steps:
        a, b = s.get("after", {}), s.get("before", {})
        if s["step"] == "answer a question":
            rows.append({"step": "Workflow: answer a question (T01)",
                         "did": f"Clicked the question '{s['question']}', typed '{s['answer']}', Add my answer" + how,
                         "guide": "The answer is added to the case file; AI re-reads the notes",
                         "happened": f"Ready after {s['secs']} s. {s['note']}. Level {b.get('level')} -> {a.get('level')}. "
                                     f"Questions now: {' | '.join(a.get('questions', [])) or '-'}",
                         "same": HAND.get("answer", {}).get("same", ""), "bad": HAND.get("answer", {}).get("bad", ""),
                         "secs": s["secs"], "comments": "Time for one 'Add my answer'. " + HAND.get("answer", {}).get("comments", "")})
        elif s["step"] == "add information":
            rows.append({"step": "Workflow: Add information (T01)",
                         "did": f"Typed '{s['added']}' in the Add box, Add" + how,
                         "guide": "New information goes into the case file; nothing already there is changed",
                         "happened": f"Ready after {s['secs']} s. {s['note']}. Level {b.get('level')} -> {a.get('level')}, "
                                     f"urgency {(b.get('p_urgent') or 0):.0%} -> {(a.get('p_urgent') or 0):.0%}. "
                                     f"To-dos now: {' | '.join(a.get('todos', [])) or '-'}",
                         "same": HAND.get("add", {}).get("same", ""), "bad": HAND.get("add", {}).get("bad", ""),
                         "shot": "screenshots/T01_after_add.png", "secs": s["secs"],
                         "comments": "Time for one 'Add'. " + HAND.get("add", {}).get("comments", "")})
        elif s["step"] == "fix numbers":
            rows.append({"step": "Workflow: Fix numbers (T01)",
                         "did": "Fix numbers: BP sys 90 -> 82, Save and re-score" + how,
                         "guide": "Re-check of 10 Nov: Observations gets 'Corrected by <name>: BP sys 82 (was 90)'",
                         "happened": f"Re-scored in {s['secs']} s. Case file line: '{s['corrected_line'] or 'none'}'. "
                                     f"Level {b.get('level')} -> {a.get('level')}, urgency {(b.get('p_urgent') or 0):.0%} -> "
                                     f"{(a.get('p_urgent') or 0):.0%}, NEWS2 {b.get('news2')} -> {a.get('news2')}",
                         "same": "yes" if "82" in s["corrected_line"] and "was 90" in s["corrected_line"] else "no",
                         "bad": "" if "82" in s["corrected_line"] else "2",
                         "shot": "screenshots/T01_after_fix.png", "secs": s["secs"],
                         "comments": "Time for one 'Save and re-score'. " + HAND.get("fix", {}).get("comments", "")})
        elif s["step"] == "chat":
            k = "chat_" + s["case"]
            rows.append({"step": f"Workflow: Chat ({s['case']})",
                         "did": f"Chat: '{s['question']}'" + how,
                         "guide": "Re-check of 10 Nov: the chat uses the case's real numbers and reasons",
                         "happened": f"Answered in {s['secs']} s: \"{' '.join(s['answer'].split())[:700]}\"",
                         "same": HAND.get(k, {}).get("same", ""), "bad": HAND.get(k, {}).get("bad", ""),
                         "shot": f"screenshots/{s['case']}_chat.png", "secs": s["secs"],
                         "comments": "Time for one chat answer. " + HAND.get(k, {}).get("comments", "")})
        elif s["step"] == "whiteboard stage + doctor":
            rows.append({"step": "Workflow: whiteboard stage + doctor (T01)",
                         "did": "As Organising doctor: stage 'Being seen', doctor 'Dr Lim'" + how,
                         "guide": "Stage and doctor change on the whiteboard",
                         "happened": s["note"], "same": "yes" if "Being seen" in s["note"] and "Dr Lim" in s["note"] else "no",
                         "shot": "screenshots/whiteboard_1_stage_doctor.png", "secs": s["secs"], "comments": ""})
        elif s["step"] == "clear with outcome":
            ok = s["on_board_after"] is False and s["clear_before_leaving_stage"][0] == 400
            rows.append({"step": "Workflow: clear a patient with an outcome (T01)",
                         "did": "Clear while 'Being seen'; then stage 'Admitted to ward', Clear with diagnosis NSTEMI, "
                                "level judged right 2" + how,
                         "guide": "Clearing needs a leaving stage first; then the patient leaves the board and the outcome is on the timeline",
                         "happened": s["note"], "same": "yes" if ok else "no", "bad": "" if ok else "2",
                         "shot": "screenshots/whiteboard_2_after_clear.png screenshots/T01_timeline.png",
                         "secs": s["secs"], "comments": ""})
    return rows + HAND_ROWS


HAND = {}                     # step -> same / bad / comments, judged by reading the workflow output
HAND_ROWS = []                # extra rows written by hand (labels re-check, Gradio whiteboard re-check)


# ---- judged by reading the 13-14 Nov results and screenshots ----
NOTE = {
    "T01": "Workflow steps (rows after R10) were done on this patient",
    "T03": "Still gets a stroke to-do (retest sheet section 5: left on purpose)",
    "T07": "Still gets a stroke to-do for a headache (retest sheet section 5: left on purpose)",
    "T09": "On the whiteboard this patient's name shows 'reading the note...' although the case is finished "
           "(no age or sex in the note), see whiteboard_2_after_clear.png",
    "P03": "Still a RED fracture mark on a normal adult X-ray (retest sheet section 5: known, image model)",
    "P05": "Case file says 'Pain 3/010' (cosmetic); child moving a sore wrist still gets a 'Trauma: primary survey' to-do",
    "P07": "Still amber 'possible pneumothorax' after it resolved (retest sheet section 5: known)",
    "P09": "The red cardiomegaly mark on this 4-year-old from 10 Nov is gone",
    "P10": "Still RED pneumonia-type shadowing + amber possible pneumothorax on a NORMAL child's X-ray (section 5: known); "
           "but now level 4, not urgent (10 Nov: level 2, urgent)",
    "P11": "Still gets a stroke to-do (section 5: left on purpose)",
    "R01": "Only difference from the key: the AI read 'brought to the emergency department' as walk-in (ambulance 0); "
           "the key's run left it out. Level and flag the same",
    "R04": "Known (retest sheet section 5): cut-off set to catch 9 in 10 urgent patients",
    "R05": "Known (retest sheet section 5)", "R06": "Known (retest sheet section 5)",
}
BAD = {"P03": "3", "P10": "3", "P07": "1", "P05": "1", "T09": "1", "R01": ""}
HAND = {
    "answer": {"same": "no", "bad": "2",
               "comments": "The answer goes into the case file and the question disappears, but the ACVPU tile and "
                           "'Consciousness not recorded' in Observations do not change: an answered number is not read "
                           "back into the tiles or the score (here level stays 2 either way)."},
    "add": {"same": "yes", "comments": "Troponin and ST depression are in the case file; level was already 2; no new "
                                       "to-do (e.g. cardiology / ACS pathway) was suggested."},
    "fix": {"comments": "FIXED (10 Nov issue): the old number is no longer left in the case file."},
    "chat_T01": {"same": "yes", "bad": "1",
                 "comments": "FIXED (10 Nov issue): uses the real numbers (BP 82 after the fix, RR 22, HR 112) and the "
                             "troponin/ECG just added. Odd extra: mentions qSOFA and sepsis risk for a chest-pain patient."},
    "chat_P05": {"same": "no", "bad": "2",
                 "comments": "Partly fixed: it now gives the real reason (BP and RR missing, safety floor), but it also "
                             "says 'NEWS2 score is 1, indicating a HIGH risk', which is wrong (NEWS2 1 is low) and "
                             "contradicts the label."},
}
HAND_ROWS = [
    {"step": "Re-check: labels (10 Nov)", "did": "Looked at the case page and charting screenshots",
     "guide": "'Questions for you to answer', 'Add my answer', '✎ Edit the case file'",
     "happened": "'Questions for you to answer' (with 'Click one, type the answer in the Add information bar above, "
                 "press Add') and '✎ Edit the case file' are shown. 'Add my answer' not checked (it appears after "
                 "clicking a question; the script sent the answer directly)",
     "same": "yes", "shot": "screenshots/T01.png", "comments": ""},
    {"step": "Re-check: Gradio whiteboard (10 Nov)", "did": "Not tested", "guide": "'Change this patient' box; "
     "'Show patients who left the ED'", "happened": "Not tested: this run used the main app only", "same": "n/a",
     "comments": "Someone on Gradio please check"},
    {"step": "Timing summary", "did": "All 40 cases timed from Submit until the case file was ready",
     "guide": "Real times for the report",
     "happened": "40 of 40 finished, no AI errors or restarts (10 Nov: 2 GPU crashes). Per case 73-161 s, mean 108 s, "
                 "72 min for all 40 (10 Nov: 77-388 s). Text cases 73-135 s, picture cases 85-161 s, real patients 77-104 s. "
                 "Save and re-score 1 s, Add 17 s, answer a question 20 s, chat 17-23 s",
     "same": "yes", "secs": "108 (mean)",
     "comments": "Laptop, NVIDIA RTX 4050 6 GB GPU, Windows 11, full AI run locally (main app, server.py). The laptop "
                 "slept between the cases (13 Nov night) and the workflow steps (14 Nov morning), so the whiteboard "
                 "shows waiting times of 11-12 h"},
]
