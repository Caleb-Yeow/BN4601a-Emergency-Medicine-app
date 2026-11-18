"""Fill the EMPTY cells of the group's feedback_form.csv from the full-AI run. Cells that already have text are never
changed. Usage: python fill_form.py <full_ai_results.jsonl> <feedback_form.csv> <expected_results.csv> [--dry]"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

res_path, form_path, exp_path = map(Path, sys.argv[1:4])
DRY = "--dry" in sys.argv
res = {}
for line in res_path.read_text(encoding="utf8").splitlines():
    if line.strip():
        r = json.loads(line)
        res[r["case"]] = r                                  # last run of a case wins
exp = {r["case"]: r for r in csv.DictReader(open(exp_path, encoding="utf-8-sig"))}
raw = form_path.read_bytes()
enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
try:
    text = raw.decode(enc)
except UnicodeDecodeError:                                  # Excel "CSV" (not UTF-8) saves in the Windows codepage
    enc, text = "cp1252", raw.decode("cp1252")
rows = list(csv.DictReader(text.splitlines()))
cols = list(rows[0])
C = dict(name=cols[0], app=cols[1], step=cols[2], did=cols[3], guide=cols[4], happened=cols[5], same=cols[6],
         bad=cols[7], shot=cols[8], secs=cols[9], comments=cols[10])
app_value = next((r[C["app"]] for r in rows if r[C["app"]].strip()), "main (full AI)")

# how bad, for rows that don't match the guide (set after reading the results)
BAD = {}
ONLY_SHOT = {"T01", "T02", "T03", "T04", "T05"}       # Caleb filled these himself: only add the screenshot
APP = "main (NVDIA GPU 6GB)"                         # Caleb's own wording in T01-T05
NOTE = {
    "T07": "Numbers right, but the full AI gives level 3 / not urgent where test mode gave level 2 / urgent",
    "T09": "Also warned 'AI value age -1 left out: not written in the note' - the number checker caught the AI's guess",
    "T10a": "T10a, T10b, T10c: identical numbers, level 2 and urgent flag",
    "T10b": "Identical to T10a", "T10c": "Identical to T10a",
    "P01": "To-do 'mechanism of injury not recorded' although the note says how it happened (same in test mode)",
    "P03": "False alarm: RED fracture mark on a normal X-ray (the guide's 1 in 4); could mislead a junior doctor",
    "P04": "Fracture found, but the full AI gives level 4 / not urgent where test mode gave level 2 / urgent",
    "P06": "Pneumothorax red as expected; pleural effusion and pneumonia-type shadowing also marked red",
    "P07": "No red pneumothorax (good) but still amber 'possible pneumothorax' after it resolved",
    "P08": "Effusion red as expected; also red shadowing and atelectasis, amber possible pneumothorax",
    "P09": "Shadowing red as expected; also red cardiomegaly",
    "P10": "False alarm on a NORMAL child's chest X-ray: RED pneumonia-type shadowing + amber possible pneumothorax",
    "P11": "Bleed found; to-do list also has 'possible poisoning' and 'possible stroke'",
    "P14": "Bleed check correctly not run",
    "P16": "Full AI gives level 2 / urgent where test mode gave level 3",
    "R01": "Full AI level 2 / urgent; test mode gave level 3",
    "R03": "Better than test mode (level 2, flagged urgent there)",
    "R04": "Test mode gave level 3 (correct); full AI over-flags it",
    "R05": "Test mode gave level 3 (correct); full AI over-flags it",
    "R06": "Test mode gave level 3 (correct); full AI over-flags it",
    "R07": "Only case in the pack given level 1",
    "R08": "Better than test mode (level 2, flagged urgent there)",
}
BAD_EXTRA = {"P03": "3", "P10": "3", "P04": "2", "T07": "1", "P07": "1"}   # 'yes' rows that still have a problem


def happened(r):
    parts = [f"{'Flagged URGENT' if r['flagged_urgent'] == 'yes' else 'Not flagged'}, level {r['level']} "
             f"({(r.get('p_urgent') or 0):.0%} chance of level 1-2), NEWS2 {r.get('news2')}",
             f"numbers: {r['numbers'] or 'none'}"]
    if r["number_warnings"]:
        parts.append(f"warnings: {r['number_warnings']}")
    if r.get("picture_sorted_as"):
        parts.append(f"picture sorted as {r['picture_sorted_as'].replace('_', ' ')}; marked: {r['picture_findings']}")
    parts.append(f"to-dos: {r['todos'] or '-'}")
    parts.append(f"questions: {r['questions'] or '-'}")
    return "; ".join(parts)


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
        return "no", BAD.get(cid, "3"), ""
    return "yes", "", ""


def vs_test_mode(cid, r):
    e = exp.get(cid)
    if not e:
        return ""
    diff = []
    if str(e["level"]) != str(r["level"]) or e["flagged_urgent"] != r["flagged_urgent"]:
        diff.append(f"level {e['level']} / urgent {e['flagged_urgent']} in test mode")
    if e["numbers"] != r["numbers"]:
        diff.append(f"numbers in test mode: {e['numbers']}")
    if e.get("picture_findings") and e["picture_findings"] != r.get("picture_findings"):
        diff.append(f"picture marks in test mode: {e['picture_findings']}")
    return ("Full AI differs from expected_results.csv (test mode): " + "; ".join(diff)) if diff else \
        "Same level, flag and numbers as expected_results.csv"


filled = 0
for row in rows:
    cid = row[C["step"]].strip()
    r = res.get(cid)
    if not r:
        continue
    e = exp.get(cid, {})
    pics = e.get("pictures", "")
    same, bad, note = judge(cid, r, row[C["guide"]])
    bad = bad or BAD_EXTRA.get(cid, "")
    shots = " ".join(f"screenshots/{s}" for s in r["screenshot"].split())
    if cid in ONLY_SHOT:
        for k, v in {C["name"]: "Caleb", C["app"]: APP, C["shot"]: shots,
                     C["happened"]: "Recreated run for the screenshot: " + happened(r)}.items():
            if not row[k].strip():
                row[k] = v
                filled += 1
        continue
    extra = "; ".join(x for x in (note, NOTE.get(cid, ""), vs_test_mode(cid, r)) if x)
    new = {C["name"]: "Caleb", C["app"]: APP,
           C["did"]: f"Inserted text from {e.get('file', cid)}" + (f", added picture {pics}" if pics else "")
                     + ", Submit (sent by a script the same way the Submit button does; full AI, 10 Nov)",
           C["happened"]: happened(r), C["same"]: same, C["bad"]: bad, C["shot"]: shots, C["secs"]: f"{r['secs'] / 60:.1f} mins ({r['secs']} s, AI time only)",
           C["comments"]: extra}
    for k, v in new.items():
        if not row[k].strip() and v:
            row[k] = v
            filled += 1

print(filled, "empty cells filled;", len(res), "cases in the results")
if DRY:
    for row in rows:
        print({k: v[:70] for k, v in row.items()})
else:
    backup = res_path.parent / "feedback_form_before_fill.csv"
    shutil.copy(form_path, backup)
    with open(form_path, "w", newline="", encoding=enc) as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("written", form_path, "| backup:", backup)
