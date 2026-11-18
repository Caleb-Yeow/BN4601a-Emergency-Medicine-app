"""Write the 13 Nov feedback form from the retest run. Always starts from the blank form inside the test-pack zip, so
running it again (after more cases finish) gives the same rows plus the new ones.
Usage: python retest_fill.py <Gradio_app> <out dir> [--dry]"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DRY = "--dry" in sys.argv
PACK = ROOT / "ED_app_test_pack"
APP = "main (NVIDIA GPU 6GB)"
COMPUTER = "Laptop, NVIDIA RTX 4050 6 GB GPU, Windows 11, full AI run locally"
sys.path.insert(0, str(Path(__file__).parent))
import retest_notes as N  # noqa: E402  (per-case comments, written after reading the results)

with zipfile.ZipFile(ROOT / "ED_app_test_pack (2).zip") as z:
    blank = z.read("ED_app_test_pack/feedback_form.csv").decode("utf-8-sig")
rows = list(csv.DictReader(io.StringIO(blank)))
cols = list(rows[0])
C = dict(name=cols[0], app=cols[1], step=cols[2], did=cols[3], guide=cols[4], happened=cols[5], same=cols[6],
         bad=cols[7], shot=cols[8], secs=cols[9], comments=cols[10])
res = {}
for line in (OUT / "results.jsonl").read_text(encoding="utf8").splitlines():
    if line.strip():
        r = json.loads(line)
        res[r["case"]] = r
pat = {f.stem: json.loads(f.read_text(encoding="utf8")) for f in (OUT / "patients").glob("*.json")}
exp = {r["case"]: r for r in csv.DictReader(open(PACK / "expected_results.csv", encoding="utf-8-sig"))}
old = {}
old_file = OUT.parent / "full_ai" / "full_ai_results.jsonl"          # the 10 Nov run, for comparison
if old_file.exists():
    old = {json.loads(l)["case"]: json.loads(l) for l in old_file.read_text(encoding="utf8").splitlines() if l.strip()}


def happened(r, p):
    d = p["decision"] or {}
    pc = f"{(d.get('p_urgent') or 0):.0%} chance of level 1-2"
    if d.get("needs_obs"):
        head = (f"Amber 'Urgent: needs vitals - {' and '.join(d.get('missing', []))} not recorded', level {p['level']} "
                f"({pc} from what is recorded)")
    else:
        head = f"{'Flagged URGENT' if d.get('flagged') else 'Not flagged'}, level {p['level']} ({pc})"
        if d.get("flagged") and p.get("reason"):
            head += f", reason shown: {p['reason']}"
    head += f", NEWS2 {d.get('news2')}"
    if p.get("child"):
        head += (" (adult score); label 'Under 16: children's urgency score; NEWS2 and level from adult models'"
                 if p.get("child_score") else " (adult score); label 'Under 16: scores from models trained on adults'")
    if p.get("raised_by"):
        head += f"; {p['raised_by']}"
    parts = [head, f"numbers: {r['numbers'] or 'none'}"]
    if r["number_warnings"]:
        parts.append(f"warnings: {r['number_warnings']}")
    if r.get("picture_sorted_as"):
        parts.append(f"picture sorted as {r['picture_sorted_as'].replace('_', ' ')}; marked: {r['picture_findings']}")
    parts.append(f"to-dos: {r['todos'] or '-'}")
    parts.append(f"questions: {r['questions'] or '-'}")
    return "; ".join(parts)


def vs_key(cid, r):
    """Differences from expected_results.csv (the group's full-AI run, 13 Nov) in what should match."""
    e = exp[cid]
    diff = []
    if str(e["level"]) != str(r["level"]) or e["flagged_urgent"] != r["flagged_urgent"]:
        diff.append(f"level {r['level']} / urgent {r['flagged_urgent']} here, level {e['level']} / urgent "
                    f"{e['flagged_urgent']} in the key")
    if e["numbers"] != r["numbers"]:
        diff.append(f"numbers in the key: {e['numbers'] or 'none'}")
    if (e.get("picture_findings") or "") != (r.get("picture_findings") or ""):
        diff.append(f"picture marks in the key: {e.get('picture_findings') or '-'}")
    return diff


def judge(cid, r, guide):
    """same_as_guide, how_bad, comment."""
    m = re.match(r"True triage level (\w+)", guide)
    if cid.startswith("R") and m:
        true = m.group(1)
        if not true.isdigit():
            return "n/a (true level unknown)", "", f"App level {r['level']}; this case report has no true triage level"
        t, a = int(true), int(r["level"])
        if a == t:
            return "yes", "", f"App level {a} = true level {t}"
        if t <= 2 < a:
            return "no", "4", f"UNDER-triage: true level {t} (urgent) but app level {a}, not flagged"
        if a < t:
            return "no", "2", f"Over-triage: app level {a}, true level {t}" + (
                " (flagged urgent although it isn't)" if r["flagged_urgent"] == "yes" else "")
        return "no", "2", f"App level {a}, true level {t} (both not urgent)"
    if r["check"] is False:
        return "no", N.BAD.get(cid, "3"), ""
    return "yes", "", ""


RC = N.rechecks(pat)
out = []
for row in rows:
    cid = row[C["step"]].strip()
    r, p = res.get(cid), pat.get(cid)
    if not r or not p:
        out.append(row)
        continue
    e = exp[cid]
    same, bad, note = judge(cid, r, row[C["guide"]])
    diff = vs_key(cid, r)
    if diff and not cid.startswith("R") and same == "yes" and cid in N.KEY_MISMATCH_IS_NO:
        same = "no"
    bad = N.BAD.get(cid, bad)
    o = old.get(cid)
    was = (f"10 Nov run: level {o['level']}, urgent {o['flagged_urgent']}, {o['secs']} s" if o else "")
    comments = "; ".join(x for x in (
        note, RC.get(cid, ""), N.NOTE.get(cid, ""),
        ("Differs from expected_results.csv: " + "; ".join(diff)) if diff else
        "Level, flag, numbers" + (" and picture marks" if e["pictures"] else "") + " same as expected_results.csv",
        was, (f"AI error on the first try, {r['retried']} restart(s); time is the successful try" if r.get("retried") else ""),
        (COMPUTER if cid == "T01" else "")) if x)
    row.update({C["name"]: "Caleb", C["app"]: APP,
                C["did"]: f"Inserted text from {e['file']}" + (f", added picture {e['pictures']}" if e["pictures"] else "")
                          + ", Submit (sent by a script the same way the Submit button does; full AI, 13 Nov)",
                C["happened"]: happened(r, p), C["same"]: same, C["bad"]: bad,
                C["shot"]: " ".join(f"screenshots/{s}" for s in r["screenshot"].split()),
                C["secs"]: str(r["secs"]), C["comments"]: comments})
    out.append(row)

for x in N.extra_rows(OUT, res, pat):                        # workflow steps and re-checks (after the 40 cases)
    out.append({**{c: "" for c in cols}, C["name"]: "Caleb", C["app"]: x.get("app", APP), C["step"]: x["step"],
                C["did"]: x["did"], C["guide"]: x["guide"], C["happened"]: x["happened"], C["same"]: x["same"],
                C["bad"]: x.get("bad", ""), C["shot"]: x.get("shot", ""), C["secs"]: str(x.get("secs", "")),
                C["comments"]: x.get("comments", "")})

done = [r for r in out if r[C["happened"]].strip()]
print(len(done), "rows filled of", len(out))
if DRY:
    for r in done:
        print("\n".join(f"  {k[:14]:14} {v[:330]}" for k, v in r.items() if v), "\n")
else:
    with open(PACK / "feedback_form.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print("written", PACK / "feedback_form.csv")
