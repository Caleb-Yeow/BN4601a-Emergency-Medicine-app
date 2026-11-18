"""Start the Gradio app in test mode on a spare port, check the page and API answer, then stop it.
Usage: python launch_check.py <BN4601A folder> <scratch dir>"""
import csv
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])

# real held-out patients: app level vs true level
key = list(csv.DictReader(open(REPO / "reasoning" / "web" / "test_notes" / "answer_key.csv", encoding="utf8")))
exp = {r["case"]: r for r in csv.DictReader(open(REPO.parent / "ED_app_test_pack" / "expected_results.csv",
                                                  encoding="utf-8-sig")) if r["case"].startswith("R")}
print("case  source   true  app  urgent(true/app)")
for i, k in enumerate(key, 1):
    e = exp[f"R{i:02d}"]
    lvl = k["true_triage_level"]
    true_urgent = ("yes" if int(lvl) <= 2 else "no") if lvl.isdigit() else "?"
    print(f"R{i:02d}   {k['source']:9} {lvl[:7]:>7}  {e['level']:>3}  {true_urgent}/{e['flagged_urgent']}")

port = 7871
log = open(SCRATCH / "gradio_launch.log", "w", encoding="utf8")
proc = subprocess.Popen([sys.executable, "-X", "utf8", str(REPO / "reasoning" / "gradio_app" / "app.py"), "--test",
                         "--port", str(port), "--board", str(SCRATCH / "launch_board.pkl")],
                        stdout=log, stderr=subprocess.STDOUT)
try:
    url, t0 = f"http://127.0.0.1:{port}", time.time()
    while time.time() - t0 < 240:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        if proc.poll() is not None:
            break
        time.sleep(2)
    print(f"\npage: HTTP {r.status_code} after {time.time() - t0:.0f} s, title found: {'<title>' in r.text}")
    info = requests.get(f"{url}/gradio_api/info", timeout=10)
    print(f"API info: HTTP {info.status_code}, {len(info.json().get('named_endpoints', {}))} named + "
          f"{len(info.json().get('unnamed_endpoints', {}))} unnamed endpoints")
finally:
    proc.terminate()
    proc.wait(timeout=30)
    log.close()
errors = [l for l in open(SCRATCH / "gradio_launch.log", encoding="utf8") if "Traceback" in l or "Error" in l]
print("errors in the app's log:", errors or "none")
