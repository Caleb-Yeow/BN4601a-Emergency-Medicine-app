"""Scan a folder for what must not go in a public repo. Prints file + rule + a masked hit; never prints a secret.
Usage: python archive_scan.py <folder> [<ktas_triage.csv>]"""
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
ktas = Path(sys.argv[2]) if len(sys.argv) > 2 else None
RULES = {
    "hf token": r"\bhf_[A-Za-z0-9]{20,}",
    "github token": r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "aws key": r"\bAKIA[0-9A-Z]{16}\b",
    "openai/anthropic key": r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "kaggle key": r"(?i)\"key\"\s*:\s*\"[0-9a-f]{32}\"|KAGGLE_KEY\s*=\s*['\"]?[0-9a-f]{20,}",
    "password/token assign": r"(?i)\b(?:password|passwd|api[_-]?key|secret|access[_-]?token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
    "my email": r"(?i)yeowcaleb@gmail\.com",
    "any email": r"[A-Za-z0-9._%+-]+@(?!users\.noreply|anthropic\.com|example\.)[A-Za-z0-9-]+\.[A-Za-z.]{2,}",
    "NUS matric": r"\bA0\d{6}[A-Z]\b",
    "NUS/HPC user path": r"(?i)/home/(?:svu/|users/[a-z]+/)?e\d{6,7}\b|\be\d{7}\b",
    "windows user path": r"C:[\\/]+Users[\\/]+Claeb",
    "KTAS raw column": r"\b(?:Chief_complain|KTAS_expert|KTAS_RN|Saturation|Length of stay_min|mistriage)\b",
}
TEXT = {".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".ipynb", ".pbs", ".sh", ".ps1", ".yml", ".yaml", ".toml", ".jsx",
        ".js", ".html", ".css", ".do", ".patch", ".mjs", ".cfg", ".ini", ".gitignore", ".gitattributes", ""}
raw_rows = set()
if ktas and ktas.exists():
    for r in csv.DictReader(open(ktas, encoding="latin-1", newline=""), delimiter=";"):
        cc = (r.get("Chief_complain") or "").strip().lower()
        if len(cc) >= 4 and r.get("SBP") and r.get("HR"):
            raw_rows.add((cc, r["SBP"].strip(), r["HR"].strip()))
    print(len(raw_rows), "KTAS fingerprints loaded (complaint + SBP + HR)")


def mask(s):
    s = s.strip().replace("\n", " ")
    return s[:4] + "…" + s[-3:] if len(s) > 10 else "…"


hits, n = {}, 0
for f in sorted(root.rglob("*")):
    if not f.is_file() or ".git" in f.parts or f.suffix.lower() not in TEXT or f.stat().st_size > 20_000_000:
        continue
    n += 1
    t = f.read_text(encoding="utf8", errors="ignore")
    rel = f.relative_to(root).as_posix()
    for name, pat in RULES.items():
        m = re.findall(pat, t)
        if m:
            first = re.search(pat, t).group(0)
            shown = first if name in ("KTAS raw column", "windows user path") else mask(first)
            hits.setdefault(name, []).append((rel, len(m), shown))
    if raw_rows:
        low = t.lower()
        fp = [x for x in raw_rows if x[0] in low and re.search(rf"\b{re.escape(x[1])}\b", low) and re.search(rf"\b{re.escape(x[2])}\b", low)]
        if len(fp) >= 3 and f.suffix.lower() in (".csv", ".jsonl", ".json", ".ipynb", ".txt"):
            hits.setdefault("KTAS row fingerprints (>=3)", []).append((rel, len(fp), "-"))
print(n, "text files scanned")
for name, items in hits.items():
    print(f"\n[{name}] {len(items)} files")
    for rel, c, s in items[:25]:
        print(f"   {c:4d}  {rel}  ({s})")
if not hits:
    print("nothing found")
