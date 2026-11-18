"""Print what a person needs to check each case: the note, the numbers the app read, the level vs truth, warnings,
case-file text, questions, to-dos, picture reading. Usage: python real40_dump.py <out dir> <case ids...|A|B>"""
import json
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
want = sys.argv[2:]
files = sorted((OUT / "patients").glob("*.json"))
for f in files:
    cid = f.stem
    if want and not any(cid == w or (len(w) == 1 and cid.startswith(w)) for w in want):
        continue
    p = json.loads(f.read_text(encoding="utf8"))
    r, k, d = p["_run"], p["_run"]["key"], p["decision"] or {}
    row = r["board_row"]
    print(f"\n######## {cid}  ({k['source']} {k['source_id']})  {r['secs']} s")
    print("NOTE:", r["note"][:1500])
    print(f"TRUE level {k['true_level']} urgent {k['true_urgent']} | APP level {p['level']} "
          f"urgent {'yes' if d.get('flagged') else 'no'} group {row.get('group')} | p {d.get('p_urgent', 0):.2f} "
          f"NEWS2 {d.get('news2')} needs_obs {d.get('needs_obs')} {d.get('missing')} | reason: {row.get('reason')!r} "
          f"| raised_by: {row.get('raised_by')!r}")
    print("FIELDS:", {k2: v for k2, v in p["fields"].items() if k2 != "chief_complaint"})
    print("COMPLAINT:", row.get("complaint"), "| who:", row.get("who"))
    print("FOUND BY:", p["found"].get("by"), "| check:", p["found"].get("check"))
    print("WARNINGS:", p["found"].get("warnings"))
    print("QUESTIONS:", p["questions"])
    print("TODOS:", [t["text"] for t in p["todos"]])
    rep = p["report"]
    keep = []
    for sec in ("Presenting complaint", "History", "Past history", "Observations", "Examination", "Investigations",
                "Imaging", "Impression", "Plan"):
        m = re.search(rf"\*\*{re.escape(sec)}[^\n]*\n(.*?)(?=\n\*\*|\Z)", rep, re.S)
        if m:
            keep.append(f"[{sec}] " + " / ".join(l.strip("- ").strip() for l in m.group(1).splitlines() if l.strip()))
    print("CASE FILE:", "\n  ".join(keep) if keep else rep[:1500])
    for n in p["notes"]:
        for i in n["images"]:
            print(f"PICTURE {i.get('name')}: sorted as {i.get('kind')} (truly {k['picture_type']})")
            print("  marks:", [("RED " if x.get("flag") else "amber ") + x["name"] for x in i.get("findings", [])
                               if x.get("flag") or x.get("possible")])
            print("  AI reading:", (i.get("text") or "")[:900])
            print("  checklist:", i.get("checklist"))
            print("  PAPER SAYS:", k["what_the_picture_shows (published caption)"][:500])
