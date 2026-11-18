"""Gradio guide 'try to break it': very long notes, capitals, typos, impossible numbers, empty note, wrong file type.
Test mode, scratch board, through the functions the buttons call. Usage: python break_tests.py <Gradio_app> <scratch>"""
import csv
import sys
import time
import traceback
from pathlib import Path

ROOT, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
REPO, PACK = ROOT / "BN4601A", ROOT / "ED_app_test_pack"
sys.argv = ["app.py", "--test", "--board", str(SCRATCH / "break_board.pkl")]
sys.path[:0] = [str(REPO / "reasoning" / "gradio_app"), str(REPO / "reasoning" / "web")]
import gradio as gr
import app as G
import make_test_pack as M

T01 = (PACK / "cases" / "T01.txt").read_text(encoding="utf8").strip()
rows = []


def wait(pid, limit=300):
    p = G.store.patients[pid]
    for _ in range(limit * 5):
        if p.status != "working" and not p.pending:
            return p
        time.sleep(0.2)
    raise RuntimeError(f"{p.bed} did not finish")


def run(text, pics=None):
    """Submit; return ('refused', msg) or ('ok', summary, seconds, header)."""
    t = time.time()
    try:
        pid = G.submit("Caleb", G.NEW, text, pics)[4]["value"]
    except gr.Error as e:
        return "refused", getattr(e, "message", str(e))
    except Exception as e:
        return "crash", f"{type(e).__name__}: {e}"
    p = wait(pid)
    s = M.summary(G.S.patient(pid))
    s["status"] = p.status
    return "ok", s, time.time() - t, G.case_view(pid)[0]


def short(s):
    return (f"level {s['level']}, urgent {s['flagged_urgent']}, numbers {s['numbers']}; "
            f"warnings: {s['number_warnings'] or 'none'}")


def log(step, did, guide, happened, same, bad="", note=""):
    rows.append({"case_or_step": step, "what_you_did": did, "what_the_guide_says": guide, "what_happened": happened,
                 "same_as_guide (yes/no)": same, "how_bad (1 cosmetic - 5 unsafe)": bad, "comments": note})
    print(f"\n[{step}] {same}\n  {happened}", flush=True)


base = run(T01)[1]
print("T01 baseline:", short(base))

# 1 very long notes
for n in (20, 80):
    text = (T01 + " ") * n + "Repeat obs: HR 120 BP 85/55."
    r = run(text)
    log(f"Very long note ({len(text):,} chars)", f"T01 pasted {n} times + a repeat set of obs at the end",
        "Works, no error", (short(r[1]) + f"; {r[2]:.1f} s") if r[0] == "ok" else f"{r[0]}: {r[1]}",
        "yes" if r[0] == "ok" and r[1]["status"] == "ready" else "no")

# 2 capitals
r = run(T01.upper())
log("Note in CAPITALS", "T01 typed all in capitals", "Same result as T01",
    short(r[1]) if r[0] == "ok" else f"{r[0]}: {r[1]}",
    "yes" if r[0] == "ok" and r[1]["level"] == base["level"] and r[1]["numbers"] == base["numbers"] else "no")

# 3 typos
typo = ("67M chset pian snice 0600, swety, radating to L arm. hr112 bp 90 / 60 rr 22 sat 94 % on air temp 37.2 "
        "pian 8/10. hx DM, HTN on metfromin. NKDA. BIBA")
r = run(typo)
log("Note with typos", f"'{typo[:60]}...' (spelling mistakes, no spaces, odd spacing)",
    "Still reads the numbers", short(r[1]) if r[0] == "ok" else f"{r[0]}: {r[1]}",
    "yes" if r[0] == "ok" and r[1]["numbers"] == base["numbers"] else "no")

# 4 impossible numbers
for label, text in [
        ("HR 400", "67M chest pain. HR 400 BP 120/80 RR 18 sats 97% RA T 37.0"),
        ("HR 1500", "67M chest pain. HR 1500 BP 120/80 RR 18 sats 97% RA T 37.0"),
        ("BP 300/20", "67M chest pain. HR 90 BP 300/20 RR 18 sats 97% RA T 37.0"),
        ("BP 60/120 (upside down)", "67M chest pain. HR 90 BP 60/120 RR 18 sats 97% RA T 37.0"),
        ("sats 150%", "67M chest pain. HR 90 BP 120/80 RR 18 sats 150% RA T 37.0"),
        ("T 50", "67M chest pain. HR 90 BP 120/80 RR 18 sats 97% RA T 50"),
        ("RR 0", "67M chest pain. HR 90 BP 120/80 RR 0 sats 97% RA T 37.0"),
        ("HR -10", "67M chest pain. HR -10 BP 120/80 RR 18 sats 97% RA T 37.0")]:
    r = run(text)
    if r[0] != "ok":
        log(f"Impossible number: {label}", text, "Ignored with a warning", f"{r[0]}: {r[1]}", "no", "3")
        continue
    warned = bool(r[1]["number_warnings"])
    log(f"Impossible number: {label}", text, "Ignored, nurse warned (like T08)", short(r[1]), "yes" if warned else "no",
        "" if warned else "3")

# 5 empty note
for label, text in [("empty", ""), ("spaces only", "    \n  "), ("one word", "pain")]:
    r = run(text)
    log(f"Empty note: {label}", f"Submit with the note {label!s} ({text!r})",
        "Refused with a message" if label != "one word" else "Works, asks for more",
        f"refused: {r[1]}" if r[0] == "refused" else (f"{r[0]}: {r[1]}" if r[0] != "ok" else
                                                      "ACCEPTED: " + short(r[1]) + f"; questions: {r[1]['questions'] or '-'}"),
        "yes" if (r[0] == "refused") == (label != "one word") else "no")

# 6 wrong file type
for label, fn, args, expect_refuse in [
        ("picture in 'Open a text file'", G.load_text_file, (str(PACK / "pictures" / "ecg.jpg"),), True),
        ("PDF in 'Open a text file'", G.load_text_file, (str(ROOT / "ED_app_gradio_guide.pdf"),), True),
        ("CSV in 'Open a text file'", G.load_text_file, (str(PACK / "expected_results.csv"),), None)]:
    try:
        out = fn(*args)
        happened, refused_ = f"accepted, {len(out or ''):,} characters put in the note box", False
    except gr.Error as e:
        happened, refused_ = f"refused: {getattr(e, 'message', e)}", True
    except Exception as e:
        happened, refused_ = f"CRASH {type(e).__name__}: {e}", None
    log(f"Wrong file type: {label}", f"opened {Path(args[0]).name}", "Refused with a message", happened,
        "yes" if refused_ is True or (expect_refuse is None and refused_ is False) else "no")

for label, pic in [("text file as a picture", PACK / "cases" / "T01.txt"),
                   ("PDF as a picture", ROOT / "ED_app_gradio_guide.pdf")]:
    r = run("67M chest pain. HR 90 BP 120/80", [str(pic)])
    log(f"Wrong file type: {label}", f"added {pic.name} in the pictures box, Submit", "Refused, or skipped with a message",
        f"refused: {r[1]}" if r[0] == "refused" else (f"{r[0]}: {r[1]}" if r[0] != "ok" else
                                                      f"accepted; status {r[1]['status']}; picture sorted as "
                                                      f"'{r[1].get('picture_sorted_as') or '-'}', findings "
                                                      f"'{r[1].get('picture_findings') or '-'}'"),
        "yes" if r[0] in ("refused", "ok") else "no")

with open(SCRATCH / "break_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\n{sum(r['same_as_guide (yes/no)'] == 'yes' for r in rows)} of {len(rows)} as expected")
