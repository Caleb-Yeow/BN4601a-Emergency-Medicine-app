"""Last QC (main 437ca2e, 14 Nov), full AI, sent the way the screens send it.
HANDOVER.md section 4 steps: T01 -> answer consciousness 'alert, GCS 15' -> Add 'trop 85, ECG ST depression V4-6' ->
chat -> Write SBAR handover; P05 + child_wrist_normal.png -> chat 'Why is this child flagged urgent?' -> whiteboard row,
pain line. Plus the new features: picture rule (P11 head-CT bleed), doctor's Change priority + review warning (T01),
three groups on the whiteboard (T04 Minor, T07 Can wait). Usage: python retest4_run.py <Gradio_app> <out dir>"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
APP, PACK = ROOT / "BN4601A_main" / "app", ROOT / "ED_app_test_pack"
SHOTS = OUT / "screenshots"
URL, WHO = "http://127.0.0.1:8000", "Caleb"
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
    return pid, wait(pid), round(time.time() - t)


def add(pid, text):
    t = time.time()
    requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": text, "who": WHO}, timeout=300).raise_for_status()
    return wait(pid), round(time.time() - t)


def row(pid):
    return next((b for b in requests.get(f"{URL}/api/board", timeout=60).json()["patients"] if b["id"] == pid), {})


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


if not up():
    subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(APP / "server.py")], cwd=APP,
                     stdout=open(OUT / "server.log", "w", encoding="utf8"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if up():
            break
        time.sleep(2)
meta = requests.get(f"{URL}/api/meta", timeout=30).json()
print("mode:", "TEST MODE" if meta.get("test") else "Full AI", flush=True)
if meta.get("test"):
    raise SystemExit("server is in test mode")

# ---------------- T01 ----------------
pid, p, secs = submit("T01")
r1 = row(pid)
rec("T01 submit", secs=secs, level=p["level"], flagged=p["decision"]["flagged"], group=r1.get("group"),
    questions=p["questions"], todos=[t["text"] for t in p["todos"]],
    note=f"level {p['level']}, group {r1.get('group')}, questions {p['questions']}")
shot(pid, "T01_1_submitted")

q = next((x for x in p["questions"] if re.search(r"conscious|gcs|avpu", x, re.I)),
         "What is the level of consciousness (AVPU/GCS)?")
before = p["fields"].get("mental")
p, secs = add(pid, f"{q} → alert, GCS 15")
obs = [l.strip() for l in p["report"].splitlines() if re.search(r"onscious|GCS", l)]
tl = [e for e in p["timeline"] if "later note" in e.get("text", "").lower()]
rec("T01 answer consciousness", secs=secs, question=q, mental=(before, p["fields"].get("mental")), report_lines=obs,
    not_recorded_left=any("consciousness not recorded" in l.lower() for l in obs), timeline=tl,
    level=p["level"], news2=p["decision"]["news2"],
    note=f"ACVPU {before} -> {p['fields'].get('mental')}; case-file lines {obs}; timeline {[e['text'] for e in tl]}")
shot(pid, "T01_2_after_GCS_answer")

p, secs = add(pid, "trop 85, ECG ST depression V4-6")
rec("T01 add trop + ECG", secs=secs, level=p["level"],
    report_lines=[l.strip() for l in p["report"].splitlines() if re.search(r"trop|st depression", l, re.I)],
    note=f"level {p['level']}")

ASK = r"(obtain|order|perform|send|check|repeat|get|request)[^.]{0,60}\b(ecg|troponin|trop)\b|\b(ecg|troponin)\b[^.]{0,30}(should be|to be) (obtained|performed|done)"
t = time.time()
qtext = "What is the most likely diagnosis and what should we do next?"
ans = requests.post(f"{URL}/api/patients/{pid}/chat", json={"text": qtext, "who": WHO}, timeout=1800).json()["answer"]
rec("T01 chat", secs=round(time.time() - t), question=qtext, answer=ans,
    asks_done_tests=[m.group(0) for m in re.finditer(ASK, ans, re.I)],
    already_done_marks=re.findall(r"\(already done[^)]*\)", ans), note=" ".join(ans.split())[:300])
shot(pid, "T01_3_chat", tab="Chat")

t = time.time()
sb = requests.post(f"{URL}/api/patients/{pid}/sbar", json={"who": WHO}, timeout=1800).json()
txt = requests.get(f"{URL}/api/patients/{pid}/sbar.txt", timeout=60)
body = sb["text"]
ai_part = body.split("**Added after arrival")[0]
rec("T01 Write SBAR handover", secs=round(time.time() - t), sbar=body, download_http=txt.status_code,
    starts=body.strip()[:60], opens_with_okay=bool(re.match(r"\W*(okay|ok|sure|here is)", body.strip(), re.I)),
    has={k: (k in ai_part.lower()) for k in ("situation", "background", "assessment", "recommendation", "troponin",
                                             "85", "st depression", "gcs 15", "90/60", "112")},
    asks_done_tests=[m.group(0) for m in re.finditer(ASK, ai_part, re.I)],
    already_done_marks=re.findall(r"\(already done[^)]*\)", ai_part), note=" ".join(body.split())[:300])
shot(pid, "T01_4_SBAR", tab="SBAR")

# doctor's Change priority, then a worse note -> review warning
ai_level = get(pid)["level"]
r = requests.put(f"{URL}/api/patients/{pid}/priority", timeout=60,
                 json={"who": "Dr Lim", "role": "Doctor", "level": 3, "reason": "pain settled, reviewed at bedside"})
nurse = requests.put(f"{URL}/api/patients/{pid}/priority", timeout=60,
                     json={"who": WHO, "role": "Triage nurse", "level": 4, "reason": "test"})
p, rw = get(pid), row(pid)
rec("T01 Change priority (doctor)", http=r.status_code, nurse_refused=(nurse.status_code, nurse.json().get("detail")),
    ai_level=ai_level, shown_level=rw.get("level"), group=rw.get("group"), override=rw.get("override"),
    reason=rw.get("reason"), review=rw.get("review"),
    timeline=[e for e in p["timeline"] if e["what"] == "Priority"],
    note=f"HTTP {r.status_code}; board level {rw.get('level')} group {rw.get('group')}; override {rw.get('override')}; "
         f"nurse tried -> {nurse.status_code} {nurse.json().get('detail')}")
shot(pid, "T01_5_priority_changed")
shot(None, "board_1_after_priority", board=True)

p, secs = add(pid, "now drowsy, responds to voice only, BP 78/50, HR 128")
rw = row(pid)
rec("T01 worse note after the change", secs=secs, ai_level_now=p["decision"]["level"], shown_level=rw.get("level"),
    review=rw.get("review"), group=rw.get("group"), fields=p["fields"],
    note=f"board level {rw.get('level')}, review: {rw.get('review')!r}")
shot(pid, "T01_6_review_warning")
shot(None, "board_2_review_warning", board=True)

# ---------------- P05 ----------------
pid5, p5, secs = submit("P05", ["child_wrist_normal.png"])
d5, r5 = p5["decision"], row(pid5)
rec("P05 submit", secs=secs, level=p5["level"], group=r5.get("group"), needs_obs=d5.get("needs_obs"),
    missing=d5.get("missing"), news2=d5.get("news2"),
    pain_lines=[l.strip() for l in p5["report"].splitlines() if "pain" in l.lower()],
    todos=[t["text"] for t in p5["todos"]], board_row={k: r5.get(k) for k in ("who", "complaint", "level", "group", "reason", "needs_obs")},
    note=f"level {p5['level']}, group {r5.get('group')}, missing {d5.get('missing')}, pain {[l.strip() for l in p5['report'].splitlines() if 'pain' in l.lower()]}")
t = time.time()
ans5 = requests.post(f"{URL}/api/patients/{pid5}/chat", json={"text": "Why is this child flagged urgent?", "who": WHO},
                     timeout=1800).json()["answer"]
rec("P05 chat", secs=round(time.time() - t), answer=ans5, note=" ".join(ans5.split())[:300])
shot(pid5, "P05_1_chat", tab="Chat")
shot(pid5, "P05_2_case_file")

# ---------------- picture rule ----------------
pid11, p11, secs = submit("P11", ["ct_head_bleed_epidural.jpg"])
r11 = row(pid11)
marks = [("RED " if f.get("flag") else "amber ") + f["name"] for n in p11["notes"] for i in n["images"]
         for f in i.get("findings", []) if f.get("flag") or f.get("possible")]
moved = [l.strip() for l in p11["report"].splitlines() if "moved up" in l.lower() or "picture" in l.lower() and "urgent" in l.lower()]
rec("P11 picture rule", secs=secs, level=p11["level"], flagged=p11["decision"]["flagged"], group=r11.get("group"),
    reason=r11.get("reason"), raised_by=r11.get("raised_by"), marks=marks, report_lines=moved,
    picture=p11["decision"].get("picture"),
    note=f"level {p11['level']}, flagged {p11['decision']['flagged']}, reason {r11.get('reason')!r}, raised_by {r11.get('raised_by')!r}, marks {marks}")
shot(pid11, "P11_1_picture_rule")
shot(pid11, "P11_2_files", tab="Files")

# ---------------- three groups ----------------
pid4, p4, s4 = submit("T04")
pid7, p7, s7 = submit("T07")
board = requests.get(f"{URL}/api/board", timeout=60).json()["patients"]
rec("Three groups", rows=[{k: b.get(k) for k in ("bed", "who", "complaint", "level", "flagged", "group", "override", "review")} for b in board],
    secs=(s4, s7), note="; ".join(f"{b['bed']} L{b['level']} {b.get('group')}" for b in board))
shot(None, "board_3_three_groups", board=True)
if B:
    B.close()
print("DONE", flush=True)
