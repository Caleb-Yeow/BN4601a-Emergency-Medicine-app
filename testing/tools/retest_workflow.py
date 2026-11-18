"""13 Nov retest, part 2 (after retest_run.py): the workflow steps the retest sheet asks for, each timed, on the
patients the case run left on the board. Sent the way the screens send them (same API calls as the buttons).
  T01: answer a question -> Add information -> Fix numbers (BP sys 90 -> 82) -> Chat -> whiteboard stage + doctor
       -> clear with an outcome
  P05: Chat ("why urgent?")
Usage: python retest_workflow.py <Gradio_app> <out dir>     -> <out>/workflow.json + screenshots"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SHOTS = ROOT / "ED_app_test_pack" / "screenshots"
URL, WHO, ROLE = "http://127.0.0.1:8000", "Caleb", "Organising doctor"
sys.path[:0] = [str(Path(__file__).parent / "cdppkgs"), str(Path(__file__).parent)]
from cdp_shot import Browser  # noqa: E402

res = {json.loads(l)["case"]: json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf8").splitlines()
       if l.strip()}
steps = []


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


def brief(p):
    d = p["decision"] or {}
    return {"level": p["level"], "flagged": d.get("flagged"), "p_urgent": d.get("p_urgent"), "news2": d.get("news2"),
            "fields": p["fields"], "todos": [t["text"] for t in p["todos"]], "questions": p["questions"]}


def step(name, **kw):
    steps.append({"step": name, **kw})
    (OUT / "workflow.json").write_text(json.dumps(steps, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{name}: {kw.get('secs', '-')} s | {str(kw.get('note', ''))[:200]}", flush=True)


B = Browser(OUT / "chrome_profile_wf")


def shot(pid, name, tab=None):
    try:
        B.case(URL, pid)
        if tab:
            B.click_tab(tab)
        return B.shot(SHOTS / f"{name}.png")
    except Exception as e:
        print("  screenshot failed:", e, flush=True)
        return ""


def shot_board(name):
    try:
        B.send("Page.navigate", url=f"{URL}/#/board")
        time.sleep(0.5)
        B.send("Page.reload")
        time.sleep(3)
        return B.shot(SHOTS / f"{name}.png")
    except Exception as e:
        print("  screenshot failed:", e, flush=True)
        return ""


pid = res["T01"]["pid"]
p0 = get(pid)
before = brief(p0)

# 1. answer one of the AI's questions (the screen puts "question -> " in the Add box; the nurse types the answer)
q = p0["questions"][0] if p0["questions"] else "Any allergies?"
t = time.time()
requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": f"{q} → alert, GCS 15", "who": WHO}, timeout=300).raise_for_status()
p = wait(pid)
step("answer a question", case="T01", secs=round(time.time() - t), question=q, answer="alert, GCS 15",
     before=before, after=brief(p), report=p["report"], status=p["status"],
     note=f"question still asked: {q in p['questions']}; mental now {p['fields'].get('mental')}")

# 2. add information
info = "trop 85 ng/L, ECG ST depression V4-6, aspirin 300 mg given"
b = brief(p)
t = time.time()
requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": info, "who": WHO}, timeout=300).raise_for_status()
p = wait(pid)
step("add information", case="T01", secs=round(time.time() - t), added=info, before=b, after=brief(p),
     report=p["report"], status=p["status"],
     note=f"troponin in case file: {'trop' in p['report'].lower()}; ST depression in case file: {'st depression' in p['report'].lower()}")
s1 = shot(pid, "T01_after_add")

# 3. Fix numbers: BP sys 90 -> 82, Save and re-score
b = brief(p)
fields = dict(p["fields"])
fields["sbp"] = 82
t = time.time()
r = requests.put(f"{URL}/api/patients/{pid}/fields", json={"fields": fields, "who": WHO}, timeout=600)
r.raise_for_status()
p = wait(pid)
line = next((l.strip() for l in p["report"].splitlines() if "Corrected by" in l), "")
step("fix numbers", case="T01", secs=round(time.time() - t), change="BP sys 90 -> 82", before=b, after=brief(p),
     report=p["report"], corrected_line=line, note=f"case file line: {line or 'NO Corrected-by line'}; level {b['level']} -> {p['level']}")
s2 = shot(pid, "T01_after_fix")

# 4. chat
for case, question in (("T01", "Why is this patient flagged urgent, and what should we rule out first?"),
                       ("P05", "Why is this child flagged urgent?")):
    cp = res[case]["pid"]
    t = time.time()
    r = requests.post(f"{URL}/api/patients/{cp}/chat", json={"text": question, "who": WHO}, timeout=1800)
    r.raise_for_status()
    step("chat", case=case, secs=round(time.time() - t), question=question, answer=r.json()["answer"],
         decision=get(cp)["decision"], fields=get(cp)["fields"], note=r.json()["answer"][:200].replace("\n", " "))
    shot(cp, f"{case}_chat", tab="Chat")

# 5. whiteboard: stage + doctor, then clear with an outcome
t = time.time()
r = requests.patch(f"{URL}/api/patients/{pid}/board", json={"who": WHO, "role": ROLE, "stage": "Being seen", "doctor": "Dr Lim"}, timeout=60)
row = next((x for x in requests.get(f"{URL}/api/board", timeout=60).json()["patients"] if x["id"] == pid), {})
step("whiteboard stage + doctor", case="T01", secs=round(time.time() - t, 1), http=r.status_code,
     note=f"board row: stage {row.get('stage')}, doctor {row.get('doctor')}")
shot_board("whiteboard_1_stage_doctor")

t = time.time()
r1 = requests.patch(f"{URL}/api/patients/{pid}/board", json={"who": WHO, "role": ROLE, "cleared": True, "diagnosis": "NSTEMI", "final_level": 2}, timeout=60)
refused = r1.status_code, (r1.json().get("detail") if r1.status_code != 200 else "")
r2 = requests.patch(f"{URL}/api/patients/{pid}/board", json={"who": WHO, "role": ROLE, "stage": "Admitted to ward"}, timeout=60)
r3 = requests.patch(f"{URL}/api/patients/{pid}/board", json={"who": WHO, "role": ROLE, "cleared": True, "diagnosis": "NSTEMI", "final_level": 2}, timeout=60)
board = requests.get(f"{URL}/api/board", timeout=60).json()
on_board = any(x["id"] == pid for x in board["patients"])
tl = [e for e in get(pid)["timeline"] if e["what"] in ("Board", "Outcome")]
step("clear with outcome", case="T01", secs=round(time.time() - t, 1),
     clear_before_leaving_stage=refused, http=[r2.status_code, r3.status_code], on_board_after=on_board,
     n_cleared=board.get("n_cleared"), timeline=tl,
     note=f"clear while 'Being seen' -> {refused}; after 'Admitted to ward' + clear: on board {on_board}, "
          f"cleared count {board.get('n_cleared')}; timeline: {' | '.join(e['text'] for e in tl[-3:])}")
shot_board("whiteboard_2_after_clear")
shot(pid, "T01_timeline", tab="Timeline")
B.close()
print("WORKFLOW DONE", flush=True)
