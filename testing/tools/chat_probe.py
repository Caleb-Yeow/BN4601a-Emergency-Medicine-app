"""Chat accuracy probe (16 Nov, main 74287be, full AI). Four patients, the same questions with checkable answers,
each answer saved with what the app knows (fields, decision, case file). Temporary patients are removed at the end.
Usage: python chat_probe.py <Gradio_app> <out dir>"""
import json
import sys
import time
from pathlib import Path

import requests

G, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
URL, WHO = "http://127.0.0.1:8000", "Caleb"
PACK, REAL = G / "ED_app_test_pack", G / "real_test_set_40"

CASES = [
    ("T01", (PACK / "cases" / "T01.txt"), None, ["trop 85, ECG ST depression V4-6"]),
    ("P05", (PACK / "cases" / "P05.txt"), PACK / "pictures" / "child_wrist_normal.png", []),
    ("A06", (REAL / "A_text_and_vitals" / "cases" / "A06.txt"), None, []),
    ("B04", (REAL / "B_text_vitals_and_picture" / "cases" / "B04.txt"),
     REAL / "B_text_vitals_and_picture" / "pictures" / "B04_ecg.jpg", []),
]
QUESTIONS = [
    ("vitals", "List this patient's vital signs exactly as recorded."),
    ("why", "Why is this patient at this triage level? Give the exact reason."),
    ("dx_next", "What is the most likely diagnosis, and what are the next 3 steps?"),
    ("imaging", "What did the imaging or ECG show?"),
    ("allergy", "Does the patient have any allergies, and what medications are they on?"),
    ("missing", "Which observations are still missing?"),
    ("memory", "What did you say the most likely diagnosis was?"),
]


def wait(pid, limit=1800):
    t = time.time()
    while time.time() - t < limit:
        p = requests.get(f"{URL}/api/patients/{pid}", timeout=60).json()
        if p["status"] != "working" and not p.get("pending"):
            return p
        time.sleep(1)
    raise RuntimeError("not finished")


made, results = [], []
try:
    for cid, note_file, pic, later in CASES:
        note = note_file.read_text(encoding="utf8").strip()
        files = [("files", (pic.name, open(pic, "rb"), "image/png" if pic.suffix == ".png" else "image/jpeg"))] if pic else None
        r = requests.post(f"{URL}/api/patients", data={"text": note, "who": WHO}, files=files, timeout=300)
        r.raise_for_status()
        pid = r.json()["id"]
        made.append(pid)
        p = wait(pid)
        for text in later:
            requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": text, "who": WHO}, timeout=300).raise_for_status()
            p = wait(pid)
        case_file = requests.get(f"{URL}/api/patients/{pid}/case_file", timeout=60).text
        rec = {"case": cid, "note": note, "later_notes": later, "picture": pic.name if pic else "",
               "fields": p["fields"], "decision": p["decision"], "level": p["level"], "report": p["report"],
               "case_file": case_file, "images": [{k: i.get(k) for k in ("kind", "text")} for n in p["notes"] for i in n["images"]],
               "chat": []}
        for qid, q in QUESTIONS:
            t = time.time()
            a = requests.post(f"{URL}/api/patients/{pid}/chat", json={"text": q, "who": WHO}, timeout=1800).json()["answer"]
            rec["chat"].append({"id": qid, "q": q, "a": a, "secs": round(time.time() - t, 1)})
            print(f"{cid} {qid:8} {round(time.time() - t):3d}s | {' '.join(a.split())[:220]}", flush=True)
        results.append(rec)
        (OUT / "chat_probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf8")
finally:
    for pid in made:                                   # leave the board as it was (the 2 demo patients)
        requests.delete(f"{URL}/api/patients/{pid}", params={"role": "Organising doctor"}, timeout=30)
    print("removed the", len(made), "probe patients; board now:",
          [b["bed"] for b in requests.get(f"{URL}/api/board", timeout=30).json()["patients"]], flush=True)
print("DONE", flush=True)
