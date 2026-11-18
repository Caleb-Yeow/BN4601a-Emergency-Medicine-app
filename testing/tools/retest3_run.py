"""HANDOVER.md section 4 retest (14 Nov, main 3c92d3d), full AI, sent the way the screens send it.
T01: Submit -> answer the consciousness question 'alert, GCS 15' (ACVPU tile + timeline) -> Add 'trop 85, ECG ST
depression V4-6' -> chat -> Write SBAR handover. P05 + child_wrist_normal.png: chat 'Why is this child flagged urgent?',
whiteboard row, case-file pain line. Usage: python retest3_run.py <Gradio_app> <out dir>"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
APP, PACK = ROOT / "BN4601A_main" / "app", ROOT / "ED_app_test_pack"
SHOTS = OUT / "screenshots"
URL, WHO, ROLE = "http://127.0.0.1:8000", "Caleb", "Organising doctor"
SHOTS.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(Path(__file__).parent / "cdppkgs"), str(Path(__file__).parent)]
from cdp_shot import Browser  # noqa: E402

log = []


def rec(step, **kw):
    log.append({"step": step, **kw})
    (OUT / "steps.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{step}: {kw.get('secs', '-')} s | {str(kw.get('note', ''))[:300]}", flush=True)


def up():
    try:
        return requests.get(f"{URL}/api/meta", timeout=5).ok
    except requests.RequestException:
        return False


def get(pid):
    return requests.get(f"{URL}/api/patients/{pid}", timeout=60).json()


def wait(pid, limit=1800):
    t = time.time()
    while time.time() - t < limit:
        p = get(pid)
        if p["status"] != "working" and not p.get("pending"):
            return p
        time.sleep(1)
    raise RuntimeError("not finished")


def submit(case, pics=()):
    note = (PACK / "cases" / f"{case}.txt").read_text(encoding="utf8").strip()
    files = [("files", (n, open(PACK / "pictures" / n, "rb"), "image/png" if n.endswith(".png") else "image/jpeg"))
             for n in pics]
    t = time.time()
    r = requests.post(f"{URL}/api/patients", data={"text": note, "who": WHO}, files=files or None, timeout=300)
    r.raise_for_status()
    pid = r.json()["id"]
    p = wait(pid)
    return pid, p, round(time.time() - t)


def add(pid, text):
    t = time.time()
    requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": text, "who": WHO}, timeout=300).raise_for_status()
    p = wait(pid)
    return p, round(time.time() - t)


B = None


def shot(pid, name, tab=None, board=False):
    global B
    try:
        if B is None:
            B = Browser(OUT / "chrome_profile")
        if board:
            B.send("Page.navigate", url=f"{URL}/#/board")
            time.sleep(0.5)
            B.send("Page.reload")
            time.sleep(3)
        else:
            B.case(URL, pid)
            if tab:
                B.click_tab(tab)
        return B.shot(SHOTS / f"{name}.png")
    except Exception as e:
        print("  screenshot failed:", e, flush=True)
        return ""


# ---- server (full AI) ----
srv = None
if not up():
    srv = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(APP / "server.py")], cwd=APP,
                           stdout=open(OUT / "server.log", "w", encoding="utf8"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if up():
            break
        time.sleep(2)
meta = requests.get(f"{URL}/api/meta", timeout=30).json()
print("mode:", "TEST MODE" if meta.get("test") else "Full AI", flush=True)
if meta.get("test"):
    raise SystemExit("server is in test mode")

# ---- T01 ----
pid, p, secs = submit("T01")
d = p["decision"]
rec("T01 submit", case="T01", secs=secs, level=p["level"], flagged=d["flagged"], fields=p["fields"],
    questions=p["questions"], todos=[t["text"] for t in p["todos"]],
    note=f"level {p['level']}, flagged {d['flagged']}, mental {p['fields'].get('mental')}, questions {p['questions']}")
shot(pid, "T01_1_submitted")

q = next((x for x in p["questions"] if "conscious" in x.lower() or "gcs" in x.lower() or "avpu" in x.lower()),
         p["questions"][0] if p["questions"] else "What is the level of consciousness (AVPU/GCS)?")
before = (p["fields"].get("mental"), p["level"], d["p_urgent"], d["news2"])
p, secs = add(pid, f"{q} → alert, GCS 15")
d = p["decision"]
tl = [e for e in p["timeline"] if "number" in (e.get("what", "") + e.get("text", "")).lower()]
obs = [l.strip() for l in p["report"].splitlines() if "onscious" in l or "GCS" in l]
rec("T01 answer consciousness", case="T01", secs=secs, question=q, answer="alert, GCS 15",
    mental_before=before[0], mental_after=p["fields"].get("mental"), level=(before[1], p["level"]),
    p_urgent=(before[2], d["p_urgent"]), news2=(before[3], d["news2"]), timeline=tl, report_lines=obs,
    found_by=p["found"].get("by", {}).get("mental"), check=p["found"].get("check"),
    warnings=p["found"].get("warnings"), questions_after=p["questions"],
    note=f"ACVPU {before[0]} -> {p['fields'].get('mental')}; level {before[1]} -> {p['level']}; NEWS2 {before[3]} -> "
         f"{d['news2']}; timeline: {' | '.join(e['what'] + ': ' + e['text'] for e in tl)[:200]}")
shot(pid, "T01_2_after_GCS_answer")
shot(pid, "T01_2b_timeline", tab="Timeline")

p, secs = add(pid, "trop 85, ECG ST depression V4-6")
rec("T01 add trop + ECG", case="T01", secs=secs, level=p["level"], todos=[t["text"] for t in p["todos"]],
    report_lines=[l.strip() for l in p["report"].splitlines() if "trop" in l.lower() or "st depression" in l.lower()],
    note=f"level {p['level']}; in case file: {[l.strip() for l in p['report'].splitlines() if 'trop' in l.lower()]}")

t = time.time()
qtext = "What is the most likely diagnosis and what should we do next?"
r = requests.post(f"{URL}/api/patients/{pid}/chat", json={"text": qtext, "who": WHO}, timeout=1800)
r.raise_for_status()
rec("T01 chat", case="T01", secs=round(time.time() - t), question=qtext, answer=r.json()["answer"],
    note=" ".join(r.json()["answer"].split())[:300])
shot(pid, "T01_3_chat", tab="Chat")

t = time.time()
r = requests.post(f"{URL}/api/patients/{pid}/sbar", json={"who": WHO}, timeout=1800)
r.raise_for_status()
sb = r.json()
txt = requests.get(f"{URL}/api/patients/{pid}/sbar.txt", timeout=60)
low = sb["text"].lower()
rec("T01 Write SBAR handover", case="T01", secs=round(time.time() - t), sbar=sb, download_http=txt.status_code,
    download_text=txt.text,
    has={k: (k in low) for k in ("situation", "background", "assessment", "recommendation", "troponin", "trop 85",
                                 "st depression", "gcs 15", "90/60", "112")},
    asks_for_troponin=any(w in low for w in ("obtain troponin", "order troponin", "check troponin", "send troponin")),
    note=" ".join(sb["text"].split())[:300])
shot(pid, "T01_4_SBAR", tab="SBAR")

# ---- P05 ----
pid5, p5, secs = submit("P05", ["child_wrist_normal.png"])
d5 = p5["decision"]
pain = [l.strip() for l in p5["report"].splitlines() if "pain" in l.lower()]
rec("P05 submit", case="P05", secs=secs, level=p5["level"], flagged=d5["flagged"], needs_obs=d5.get("needs_obs"),
    missing=d5.get("missing"), news2=d5.get("news2"), pain_lines=pain, todos=[t["text"] for t in p5["todos"]],
    marks=[f["name"] for n in p5["notes"] for i in n["images"] for f in i.get("findings", []) if f.get("flag") or f.get("possible")],
    note=f"level {p5['level']}, needs_obs {d5.get('needs_obs')} {d5.get('missing')}; pain lines: {pain}")
t = time.time()
r = requests.post(f"{URL}/api/patients/{pid5}/chat", json={"text": "Why is this child flagged urgent?", "who": WHO},
                  timeout=1800)
r.raise_for_status()
ans = r.json()["answer"]
rec("P05 chat", case="P05", secs=round(time.time() - t), question="Why is this child flagged urgent?", answer=ans,
    says_high_risk_news2=("high risk" in ans.lower() and "news2" in ans.lower() and "low" not in ans.lower()),
    note=" ".join(ans.split())[:300])
shot(pid5, "P05_1_chat", tab="Chat")
shot(pid5, "P05_2_case_file")

# ---- whiteboard rows ----
board = requests.get(f"{URL}/api/board", timeout=60).json()["patients"]
rows = {b["id"]: b for b in board}
rec("whiteboard", rows={k: v for k, v in rows.items() if k in (pid, pid5)}, all_rows=board,
    note=f"P05 row: {json.dumps(rows.get(pid5, {}), ensure_ascii=False)[:250]}")
shot(None, "whiteboard", board=True)

# T09-style row (no age or sex) to check the "age/sex not recorded" fix on a finished patient
pid9, p9, secs = submit("T09")
row9 = next((b for b in requests.get(f"{URL}/api/board", timeout=60).json()["patients"] if b["id"] == pid9), {})
rec("T09 whiteboard row", case="T09", secs=secs, row=row9, note=json.dumps(row9, ensure_ascii=False)[:300])
shot(None, "whiteboard_with_T09", board=True)
if B:
    B.close()
print("DONE", flush=True)
