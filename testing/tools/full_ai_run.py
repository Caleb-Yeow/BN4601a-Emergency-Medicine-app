"""Run test-pack cases through the running FULL-AI main app (server.py, no --test), the way Submit sends them,
check each against the guide, and screenshot the patient's case page with headless Chrome.
Usage: python full_ai_run.py <Gradio_app> <out dir> [case ids ...]   (results appended to <out>/full_ai_results.jsonl)"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ONLY = set(sys.argv[3:])
PACK, WEB = ROOT / "ED_app_test_pack", ROOT / "BN4601A" / "reasoning" / "web"
SHOTS = PACK / "screenshots"
URL, WHO = "http://127.0.0.1:8000", "Caleb"
CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
sys.path.insert(0, str(WEB))
import make_test_pack as M  # noqa: E402

SHOTS.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
checks = {c[0]: c[5] for c in M.CASES}
expected = {r["case"]: r for r in csv.DictReader(open(PACK / "expected_results.csv", encoding="utf-8-sig"))}


def wait(pid, limit=1200):
    t = time.time()
    while time.time() - t < limit:
        p = requests.get(f"{URL}/api/patients/{pid}", timeout=30).json()
        if p["status"] != "working" and not p.get("pending"):
            return p
        time.sleep(1)
    raise RuntimeError(f"{pid} not finished after {limit} s")


sys.path[:0] = [str(Path(__file__).parent / "cdppkgs"), str(Path(__file__).parent)]
from cdp_shot import Browser  # noqa: E402

B = Browser(OUT / "chrome_profile")


def shot(pid, name, pictures, retry=True):
    """Case page; for picture cases also the Files tab (the picture with its marks and the AI's reading)."""
    global B
    try:
        B.case(URL, pid)
        names = [B.shot(SHOTS / f"{name}.png")]
        if pictures and B.click_tab("Files"):
            names.append(B.shot(SHOTS / f"{name}_files.png"))
        return " ".join(names)
    except Exception as e:
        print(f"  screenshot failed for {name}: {e}" + (" - restarting Chrome" if retry else ""), flush=True)
        try:
            B.close()
        except Exception:
            pass
        if not retry:
            return ""
        B = Browser(OUT / "chrome_profile")
        return shot(pid, name, pictures, retry=False)


SERVER = None


def up():
    try:
        return requests.get(f"{URL}/api/meta", timeout=5).ok
    except requests.RequestException:
        return False


def start_server():
    """Start the full-AI server (unless one is already answering) and wait until it's ready."""
    global SERVER
    if up():
        return
    log = open(OUT / f"server_{time.strftime('%H%M%S')}.log", "w", encoding="utf8")
    SERVER = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(WEB / "server.py")], cwd=WEB,
                              stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if up():
            print("  server ready", flush=True)
            return
        if SERVER.poll() is not None:
            raise SystemExit("server exited while starting")
        time.sleep(2)
    raise SystemExit("server didn't start in 10 min")


def restart_server():
    global SERVER
    print("  restarting the server (GPU error)", flush=True)
    if SERVER and SERVER.poll() is None:
        SERVER.terminate()
        SERVER.wait(60)
    else:                                         # a server this script didn't start: stop whatever holds port 8000
        subprocess.run(["powershell", "-NoProfile", "-c", "Get-NetTCPConnection -LocalPort 8000 -State Listen "
                        "-ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"])
    time.sleep(10)
    SERVER = None
    start_server()


def clear_errors():
    for q in requests.get(f"{URL}/api/board", timeout=30).json()["patients"]:
        if q.get("status") == "error":
            requests.delete(f"{URL}/api/patients/{q['id']}", params={"role": "Organising doctor"}, timeout=30)


def submit(e):
    note = (PACK / e["file"]).read_text(encoding="utf8").strip()
    files = [("files", (Path(n).name, open(PACK / n, "rb"), "image/png" if n.endswith(".png") else "image/jpeg"))
             for n in e["pictures"].split()]
    t = time.time()
    r = requests.post(f"{URL}/api/patients", data={"text": note, "who": WHO}, files=files or None, timeout=300)
    r.raise_for_status()
    pid = r.json()["id"]
    return pid, wait(pid), round(time.time() - t)


start_server()
clear_errors()
done = set()
res_file = OUT / "full_ai_results.jsonl"
if res_file.exists():
    done = {json.loads(l)["case"] for l in res_file.read_text(encoding="utf8").splitlines() if l.strip()}
for cid, e in expected.items():
    if (ONLY and cid not in ONLY) or (not ONLY and cid in done):
        continue
    pics = e["pictures"].split()
    for attempt in (1, 2):
        try:
            pid, p, secs = submit(e)
        except requests.RequestException as err:      # server died mid-case
            print(f"  {cid}: server not answering ({err})", flush=True)
            p = {"status": "error"}
        if p["status"] != "error":
            break
        print(f"{cid} AI ERROR on attempt {attempt}", flush=True)
        restart_server()
        clear_errors()
    if p["status"] == "error":
        print(f"{cid} failed twice - stopping", flush=True)
        break
    s = M.summary(p)
    chk = checks.get(cid)
    ok = None if chk is None else bool(chk(p))
    row = {"case": cid, "pid": pid, "bed": p.get("bed"), "secs": secs, "check": ok, "status": p["status"],
           "p_urgent": (p["decision"] or {}).get("p_urgent"), "news2": (p["decision"] or {}).get("news2"),
           "report": p.get("report", ""), **s, "screenshot": shot(pid, cid, pics)}
    with open(res_file, "a", encoding="utf8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{cid:5} {secs:4d}s check {ok} level {s['level']} urgent {s['flagged_urgent']} "
          f"{s.get('picture_sorted_as', '')} {s.get('picture_findings', '')} shot {row['screenshot'] or 'MISSING'}",
          flush=True)
B.close()
print("ALL DONE", flush=True)
