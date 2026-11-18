"""Run the group test pack through the Gradio app's own screen functions (test mode) and compare with
expected_results.csv. Usage: python run_test_pack_gradio.py <BN4601A folder> <ED_app_test_pack folder> <scratch dir>"""
import csv
import sys
import time
from pathlib import Path

REPO, PACK, SCRATCH = (Path(a) for a in sys.argv[1:4])
sys.argv = ["app.py", "--test", "--board", str(SCRATCH / "board.pkl")]
sys.path[:0] = [str(REPO / "reasoning" / "gradio_app"), str(REPO / "reasoning" / "web")]
import app as G                     # the Gradio app (not launched: launch is under __main__)
import make_test_pack as M          # the group's own checks and summary formatting

checks = {c[0]: c[5] for c in M.CASES}
expected = {r["case"]: r for r in csv.DictReader(open(PACK / "expected_results.csv", encoding="utf-8-sig"))}
COMPARE = ["level", "flagged_urgent", "numbers", "todos", "questions", "picture_sorted_as", "picture_findings"]

rows, t0 = [], time.time()
for cid, exp in expected.items():
    text = (PACK / exp["file"]).read_text(encoding="utf8").strip()
    pics = [str(PACK / p) for p in exp["pictures"].split()]
    out = G.submit("Test nurse", G.NEW, text, pics)
    pid = out[4]["value"]
    p = G.store.patients[pid]
    for _ in range(600):
        if p.status != "working" and not p.pending:
            break
        time.sleep(0.2)
    screen = G.case_view(pid)                       # what the Case file tab would show
    d = G.S.patient(pid)
    got = M.summary(d)
    diffs = [k for k in COMPARE if str(got.get(k, "") or "") != str(exp.get(k, "") or "")]
    check = checks.get(cid)
    guide = "record" if check is None else ("PASS" if check(d) else "FAIL")
    rows.append((cid, p.status, guide, diffs, got, exp, screen[0].splitlines()[0]))
    print(f"{cid:5} status={p.status:6} guide={guide:6} level {got['level']} (exp {exp['level']}) "
          f"flag {got['flagged_urgent']} (exp {exp['flagged_urgent']}) "
          f"{'MATCH' if not diffs else 'DIFF: ' + ', '.join(diffs)}", flush=True)

t10 = {(r[4]["numbers"], r[4]["level"], r[4]["flagged_urgent"]) for r in rows if r[0].startswith("T10")}
print(f"\nT10a/b/c consistent: {len(t10) == 1}")
print(f"{len(rows)} cases in {time.time() - t0:.0f} s | errors: {sum(r[1] == 'error' for r in rows)} | "
      f"guide checks FAIL: {[r[0] for r in rows if r[2] == 'FAIL']} | "
      f"exact match with expected_results.csv: {sum(not r[3] for r in rows)}/{len(rows)}")
for cid, _, _, diffs, got, exp, _ in rows:
    for k in diffs:
        print(f"\n[{cid}] {k}\n   expected: {exp.get(k)}\n   got:      {got.get(k)}")
real = [(r[0], r[4]["level"]) for r in rows if r[0].startswith("R")]
print("\nreal held-out patients (app level):", real)
