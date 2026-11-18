"""Run my_test_cases through the Gradio app's screen functions (test mode) and record what the app shows.
Usage: python run_my_cases.py <Gradio_app folder> <scratch dir>"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
REPO, PICS, MINE = ROOT / "BN4601A", ROOT / "ED_app_test_pack" / "pictures", ROOT / "my_test_cases"
sys.argv = ["app.py", "--test", "--board", str(SCRATCH / "my_board.pkl")]
sys.path[:0] = [str(REPO / "reasoning" / "gradio_app"), str(REPO / "reasoning" / "web")]
import app as G
import make_test_pack as M

CASES = [("C01", "Confused, feverish, low BP (temperature in °F)", []),
         ("C02", "Chest pain with normal vitals", []),
         ("C03", "Collapsed, unresponsive, no vitals (safety rule)", []),
         ("C04", "Impossible heart rate typo (HR 1500)", []),
         ("C05", "Minor ankle injury, X-ray without fracture", ["adult_leg_no_fracture.jpg"]),
         ("C06", "Head injury, CT with bleed", ["ct_head_bleed_epidural.jpg"])]


def wait(p):
    for _ in range(900):
        if p.status != "working" and not p.pending:
            return
        time.sleep(0.2)
    raise SystemExit(f"{p.bed} did not finish")


def record(cid, title, pics, pid):
    d = G.S.patient(pid)
    dec = d["decision"] or {}
    s = M.summary(d)
    header = G.case_view(pid)[0]
    raised = next((l for l in header.splitlines() if l.startswith("⬆")), "")
    return {"case": cid, "title": title, "note_file": f"cases/{cid}.txt",
            "pictures": " ".join(f"ED_app_test_pack/pictures/{p}" for p in pics),
            "level": s["level"], "flagged_urgent": s["flagged_urgent"],
            "p_urgent": f"{dec.get('p_urgent', 0):.0%}", "news2": dec.get("news2"),
            "raised_by_safety_floor": raised.lstrip("⬆ "), "numbers": s["numbers"],
            "number_warnings": s["number_warnings"], "todos": s["todos"],
            "picture_sorted_as": s.get("picture_sorted_as", ""), "picture_findings": s.get("picture_findings", "")}


rows = []
for cid, title, pics in CASES:
    text = (MINE / "cases" / f"{cid}.txt").read_text(encoding="utf8").strip()
    pid = G.submit("Test nurse", G.NEW, text, [str(PICS / p) for p in pics])[4]["value"]
    wait(G.store.patients[pid])
    rows.append(record(cid, title, pics, pid))
    if cid == "C02":                                     # follow-up: add a result to the same patient
        G.add_info("Test nurse", pid, (MINE / "cases" / "C02_followup.txt").read_text(encoding="utf8").strip(), None)
        wait(G.store.patients[pid])
        rows.append(record("C02+", "C02 after adding troponin + ECG (Add information)", [], pid))
        rows[-1]["note_file"] = "cases/C02_followup.txt (added to C02)"

with open(MINE / "my_expected_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(rows, indent=1, ensure_ascii=False))
