"""Gradio guide section 5, scripted: cases T01/T02/P01, every button once (through the functions the buttons call),
and the 'try to break it' list. Test mode, scratch board. Usage: python gradio_walkthrough.py <Gradio_app> <scratch>"""
import csv
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
REPO, PACK = ROOT / "BN4601A", ROOT / "ED_app_test_pack"
sys.argv = ["app.py", "--test", "--board", str(SCRATCH / "walk_board.pkl")]
sys.path[:0] = [str(REPO / "reasoning" / "gradio_app"), str(REPO / "reasoning" / "web")]
import gradio as gr
import app as G
import make_test_pack as M

NAME, LEAD = "Caleb", "Organising doctor"
expected = {r["case"]: r for r in csv.DictReader(open(PACK / "expected_results.csv", encoding="utf-8-sig"))}
results = []


def log(step, did, guide, happened, ok, bad="", note=""):
    results.append({"case_or_step": step, "what_you_did": did, "what_the_guide_says": guide,
                    "what_happened": happened, "same_as_guide (yes/no)": "yes" if ok else "no",
                    "how_bad (1 cosmetic - 5 unsafe)": bad, "comments": note})
    print(f"{'OK  ' if ok else 'DIFF'} {step}: {happened}", flush=True)


def wait(pid, limit=180):
    p = G.store.patients[pid]
    for _ in range(limit * 5):
        if p.status != "working" and not p.pending:
            return p
        time.sleep(0.2)
    raise RuntimeError(f"{p.bed} did not finish")


def refused(fn, *a):
    """True + message if the action is refused with a Gradio warning/error (what the screen shows)."""
    try:
        fn(*a)
        return False, "accepted"
    except gr.Error as e:
        return True, getattr(e, "message", str(e))


# ------------------------------------------------------------- cases via "Open a text file" + Submit
pids = {}
for cid in ("T01", "T02", "P01"):
    exp = expected[cid]
    t = time.time()
    text = G.load_text_file(str(PACK / exp["file"]))                       # 📄 Open a text file
    pics = [str(PACK / x) for x in exp["pictures"].split()]
    pid = G.submit(NAME, G.NEW, text, pics)[4]["value"]                   # Submit
    p = wait(pid)
    d = G.S.patient(pid)
    s = M.summary(d)
    pids[cid] = pid
    check = {c[0]: c[5] for c in M.CASES}[cid](d)
    same = s["level"] == int(exp["level"]) and s["numbers"] == exp["numbers"] and check
    extra = ""
    if cid == "P01":
        labels = [lbl for _, lbl in G.case_view(pid)[5]]
        extra = f"; Files tab shows: {', '.join(labels)}"
    log(cid, f"Opened cases/{cid}.txt with 'Open a text file'" + (f", added {exp['pictures']}" if pics else "") + ", Submit",
        exp["what_the_guide_says"],
        f"level {s['level']}, flagged {s['flagged_urgent']}, numbers {s['numbers']}; to-dos: {s['todos'] or '-'}; "
        f"questions: {s['questions'] or '-'}" + (f"; picture: {s['picture_sorted_as']}, {s['picture_findings']}" if pics else "")
        + extra + f" ({time.time() - t:.1f} s)", same, "" if same else "?")

# ------------------------------------------------------------- every button once (on T01)
pid = pids["T01"]
p = G.store.patients[pid]
try:
    q = p.questions[0] if p.questions else "Any allergies?"
    line = G.pick_question(q) + "no known allergies"                      # click a question -> answer -> Add
    G.add_info(NAME, pid, line, None)
    p = wait(pid)
    ok = any("no known allergies" in n["text"] for n in p.notes)
    log("Button: question → answer → Add", f"picked '{q}', typed an answer, Add", "Answer added to the notes",
        "answer is in the notes and the case file was rebuilt" if ok else "answer missing", ok)

    G.set_todos(NAME, LEAD, pid, ["0"])                                   # tick a to-do
    ticked = p.todos[0]["done"]
    G.add_todo(NAME, LEAD, pid, "Repeat ECG in 30 min")                   # add a to-do
    added = any(t["text"] == "Repeat ECG in 30 min" and not t.get("ai") for t in p.todos)
    log("Button: tick and add a to-do", "ticked the first to-do, added 'Repeat ECG in 30 min'", "Ticked; own to-do added",
        f"ticked: {ticked}; added (not marked AI): {added}", ticked and added)

    table = G.fix_form(pid)[0]["value"]                                   # Fix numbers
    for row in table:
        if row[0] == "HR":
            row[1] = "130"
    G.save_fix(NAME, pid, table)
    p = wait(pid)
    log("Button: Fix numbers", "changed HR 112 -> 130, saved", "Saved and re-scored, logged on the timeline",
        f"HR now {p.fields.get('hr')}, level {G.S.patient(pid)['level']}, timeline entries {len(p.visit.timeline)}",
        p.fields.get("hr") == 130)

    report = G.start_edit(pid)[0]["value"]                                # Edit -> Save
    G.save_edit(NAME, pid, report + "\nEdited by Caleb: patient anxious.")
    ok = p.report_by == NAME and "patient anxious" in p.report
    log("Button: Edit and Save", "Edit, added a line, Save", "Case file keeps the edit and who made it",
        f"report_by = {p.report_by}, edit kept: {'patient anxious' in p.report}", ok)

    path = G._case_file_path(p)                                           # Download
    body = Path(path).read_text(encoding="utf-8")
    ok = "patient anxious" in body and "no known allergies" in body
    log("Button: Download", "Download the whole case file", "One text file with the case file, notes and timeline",
        f"{Path(path).name}, {len(body):,} characters, contains the edit and the added answer: {ok}", ok)

    G.ask_chat(NAME, pid, "What else should we rule out?")                # Chat
    answered = len(p.chat) >= 2 and p.chat[-1]["role"] != "user"
    log("Button: Chat", "asked 'What else should we rule out?'", "Answered (placeholder in test mode)",
        f"answer: {p.chat[-1]['text'][:90] if p.chat else '-'}", answered, "", "test mode answer is a placeholder")

    G.board_update(NAME, LEAD, pid, "Being seen", "Dr Lim", False)        # whiteboard stage + doctor
    ok = p.stage == "Being seen" and p.doctor == "Dr Lim"
    log("Button: whiteboard stage + doctor", "Organising doctor set stage 'Being seen', doctor 'Dr Lim'",
        "Stage and doctor change on the board", f"stage {p.stage}, doctor {p.doctor}", ok)

    G.board_update(NAME, LEAD, pid, "Admitted to ward", "Dr Lim", False)   # Clear and Put back
    G.board_clear(NAME, LEAD, pid, "NSTEMI", "2", False)
    on_board = any(r[1] == p.bed for r in G.board_view(False)[0])
    in_left = any(r[1] == p.bed for r in G.board_view(True)[0])
    G.board_put_back(NAME, LEAD, pid)
    back = any(r[1] == p.bed for r in G.board_view(False)[0])
    log("Button: Clear and Put back", "stage 'Admitted to ward', Clear (diagnosis NSTEMI, level 2), then Put back",
        "Gone from the board, listed under Left the ED, Put back returns it",
        f"after Clear: on board {on_board}, in Left the ED {in_left}; after Put back: on board {back}",
        (not on_board) and in_left and back)
except Exception as e:                                                    # record, don't stop the run
    log("Buttons", "every button once", "", f"CRASH {type(e).__name__}: {e}", False, "4", traceback.format_exc()[-300:])

# ------------------------------------------------------------- try to break it
ok, msg = refused(G.submit, "", G.NEW, "50M cough", None)
log("Break: Submit without a name", "Submit with the name box empty", "Refused with a message", msg, ok)

ok, msg = refused(G.load_text_file, str(PACK / "pictures" / "ecg.jpg"))
log("Break: picture in 'Open a text file'", "opened pictures/ecg.jpg with 'Open a text file'", "Refused with a message",
    msg, ok)

errors, made = [], []
def tab(n):
    try:
        for i in range(3):
            pid = G.submit(f"Tab{n}", G.NEW, f"{40 + i}M chest pain from tab {n}, HR {90 + i} BP 120/80", None)[4]["value"]
            made.append(pid)
            G.board_view(False)
        G.board_update(f"Tab{n}", LEAD, pids["T02"], "Results pending", f"Dr Tab{n}", False)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
threads = [threading.Thread(target=tab, args=(n,)) for n in (1, 2)]
[t.start() for t in threads]
[t.join() for t in threads]
for pid in made:
    wait(pid)
beds = [G.store.patients[x].bed for x in made]
t02 = G.store.patients[pids["T02"]]
ok = not errors and len(set(beds)) == len(beds) == 6
log("Break: two tabs at once", "two users submitted 3 patients each and edited the same board row at the same time "
    "(simulated in one process, not two real browser tabs)", "Both work; board stays consistent",
    f"errors: {errors or 'none'}; 6 patients, bed labels {beds}; T02 doctor ends as '{t02.doctor}' (last writer wins)",
    ok, "" if ok else "3", "duplicate bed labels would confuse the board" if len(set(beds)) != len(beds) else "")

messy = ("pt is a 58 yo M ?? c/o SOB++ & chest disconfort x3/7, worse on exertion, also some ankle swelling \U0001F915. "
         "Wife says hes been 'not himself'. PMH: HTN, T2DM (on metformin 500mg BD), ?CKD3, prev MI 2019 (stent x2). "
         "Allergic: penicillin (rash). obs @ triage: HR 104 irreg, BP 158 / 94, RR 24, SpO2 92 % RA, temp 37.4C. "
         "earlier at GP: HR 98, BP 150/90. smokes 20/day. lives alone, independent. " * 4
         + "ECG pending. Ns to repeat obs. Nurse: ")
t = time.time()
pid = G.submit(NAME, G.NEW, messy, None)[4]["value"]
p = wait(pid)
s = M.summary(G.S.patient(pid))
log("Break: long, messy note", f"~{len(messy):,}-character note with typos, an emoji, repeated text, two sets of vitals",
    "Reads sensible numbers, no error",
    f"status {p.status}, level {s['level']}, numbers {s['numbers']}; warnings: {s['number_warnings'] or '-'} "
    f"({time.time() - t:.1f} s)", p.status == "ready")

with open(SCRATCH / "walkthrough_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)
print(f"\n{sum(r['same_as_guide (yes/no)'] == 'yes' for r in results)} of {len(results)} as the guide says")
