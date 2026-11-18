"""Real held-out test set (40 patients, 16 Nov) through the running FULL-AI main app, sent the way Submit sends it
(set B with its picture). Saves every patient's full record for checking, plus screenshots.
Usage: python real40_run.py <test set folder> <out dir>   (finished cases are skipped on a re-run)"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

T, OUT = Path(sys.argv[1]), Path(sys.argv[2])
APP = T.parent / "BN4601A_main" / "app"
SHOTS = OUT / "screenshots"
URL, WHO = "http://127.0.0.1:8000", "Caleb"
(OUT / "patients").mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(exist_ok=True)
sys.path[:0] = [str(Path(__file__).parent / "cdppkgs"), str(Path(__file__).parent)]
from cdp_shot import Browser  # noqa: E402

key = list(csv.DictReader(open(T / "answer_key.csv", encoding="utf-8-sig")))
B = None


def up():
    try:
        return requests.get(f"{URL}/api/meta", timeout=5).ok
    except requests.RequestException:
        return False


def wait(pid, limit=1800):
    t = time.time()
    while time.time() - t < limit:
        p = requests.get(f"{URL}/api/patients/{pid}", timeout=60).json()
        if p["status"] != "working" and not p.get("pending"):
            return p
        time.sleep(1)
    raise RuntimeError("not finished")


def shot(pid, name, picture):
    global B
    try:
        if B is None:
            B = Browser(OUT / "chrome_profile")
        B.case(URL, pid)
        names = [B.shot(SHOTS / f"{name}.png")]
        if picture and B.click_tab("Files"):
            names.append(B.shot(SHOTS / f"{name}_files.png"))
        return " ".join(names)
    except Exception as e:
        print(f"  screenshot failed for {name}: {e}", flush=True)
        try:
            B.close()
        except Exception:
            pass
        B = None
        return ""


def restart():
    print("  restarting the server (AI error)", flush=True)
    subprocess.run(["powershell", "-NoProfile", "-c", "Get-NetTCPConnection -LocalPort 8000 -State Listen "
                    "-ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"])
    time.sleep(10)
    subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(APP / "server.py")], cwd=APP,
                     stdout=open(OUT / f"server_{time.strftime('%H%M%S')}.log", "w", encoding="utf8"),
                     stderr=subprocess.STDOUT)
    for _ in range(300):
        if up():
            return
        time.sleep(2)
    raise SystemExit("server did not come back")


if not up():
    raise SystemExit("start the app first (python server.py)")
if requests.get(f"{URL}/api/meta", timeout=30).json().get("test"):
    raise SystemExit("server is in test mode")
done = {f.stem for f in (OUT / "patients").glob("*.json")}
for k in key:
    cid = k["case"]
    if cid in done:
        continue
    folder = T / ("A_text_and_vitals" if cid.startswith("A") else "B_text_vitals_and_picture")
    note = (folder / "cases" / f"{cid}.txt").read_text(encoding="utf8").strip()
    pic = k["picture"].strip()
    for attempt in (1, 2):
        files = ([("files", (pic, open(folder / "pictures" / pic, "rb"), "image/png" if pic.endswith(".png") else "image/jpeg"))]
                 if pic else None)
        t = time.time()
        try:
            r = requests.post(f"{URL}/api/patients", data={"text": note, "who": WHO}, files=files, timeout=300)
            r.raise_for_status()
            pid = r.json()["id"]
            p = wait(pid)
        except (requests.RequestException, RuntimeError) as e:
            print(f"  {cid}: {e}", flush=True)
            p = {"status": "error"}
        secs = round(time.time() - t)
        if p["status"] != "error":
            break
        print(f"{cid} AI ERROR on attempt {attempt}", flush=True)
        restart()
    if p["status"] == "error":
        print(f"{cid} failed twice - stopping", flush=True)
        break
    board = next((b for b in requests.get(f"{URL}/api/board", timeout=60).json()["patients"] if b["id"] == pid), {})
    p["_run"] = {"case": cid, "secs": secs, "note": note, "picture": pic, "board_row": board,
                 "screenshot": shot(pid, cid, pic), "key": k}
    (OUT / "patients" / f"{cid}.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf8")
    d = p["decision"] or {}
    img = [i for n in p["notes"] for i in n["images"]]
    marks = "; ".join(("RED " if f.get("flag") else "amber ") + f["name"] for i in img for f in i.get("findings", [])
                      if f.get("flag") or f.get("possible")) if img else ""
    print(f"{cid:4} {secs:4d}s level {p['level']} (true {k['true_level']}) urgent {'yes' if d.get('flagged') else 'no'} "
          f"(true {k['true_urgent']}) group {board.get('group')}"
          + (f" | picture sorted as {img[0].get('kind')} (is {k['picture_type']}); marks: {marks or 'none'}" if img else ""),
          flush=True)
if B:
    B.close()
print("ALL DONE", flush=True)
