#!/usr/bin/env python3
"""Whole-job progress bar and time-left estimate for an ED-VLM PBS run, read from its live log.

  python3 progress.py                    # newest log in $ED_VLM_WORKDIR/pbs_logs (default /scratch/<NUS-ID>/ed_vlm)
  python3 progress.py path/to/job.log    # a specific log (or the PBS .o file)
  python3 progress.py --watch 60         # refresh every 60 s (Ctrl+C to quit)

Standard library only, so it runs on the login node without activating the venv.
"""
import argparse, glob, json, math, os, re, sys, time
from datetime import datetime, timedelta

# what ed_vlm_hpc_vanda.pbs passes; run_config.json in the run dir overrides these when it can be read
DEFAULTS = dict(per_device_bs=2, grad_accum=16, n_eval_gen=250, gen_bs=8, max_test_examples=400, max_val_examples=300,
                eval_minutes=150, train_minutes=1080, eval_no_note=True, patience=3)
GEN_SEC_PER_IMAGE = 4.0      # last run on the A40: ~3.3 s per test image (findings + assessment); used until measured
REPORT_SEC = 90

TS = re.compile(r"\[(\d\d):(\d\d):(\d\d)\] ")
BAR = re.compile(r"(?P<desc>[^:|\[\]]+?): +\d+%\|[^|]*\| *(?P<n>\d+)/(?P<tot>\d+) \[(?P<el>[\d:]+)<(?P<rem>[\d:?]+)"
                 r"(?:, *(?P<rate>[\d.]+|\?)(?P<unit>s/it|it/s))?")


def hms(t):
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))


def dur(sec):
    sec = max(0, int(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m" if m else f"{sec}s"


def bar(frac, width):
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    return "█" * full + ("▌" if frac * width - full >= 0.5 else "") + "░" * (width - full - (frac * width - full >= 0.5))


def newest_log():
    root = os.environ.get("ED_VLM_WORKDIR") or "/scratch/<NUS-ID>/ed_vlm"
    logs = glob.glob(os.path.join(root, "pbs_logs", "job-*.log"))
    if not logs:                                  # not on the HPC: a downloaded log next to this script or in the cwd
        here = os.path.dirname(os.path.abspath(__file__))
        logs = [f for d in {here, os.getcwd()} for pat in ("job-*.log", "ed_vlm.o*") for f in glob.glob(os.path.join(d, pat))]
    if not logs:
        sys.exit(f"No job log found (looked in {root}/pbs_logs and next to this script).\n"
                 "On the HPC: is the job still queued? (qstat -u $USER)\n"
                 "On your laptop: pass the log, e.g.\n"
                 "  python progress.py path\\to\\job-1234567.log\n"
                 "  python progress.py https://github.com/<user>/<repo>/blob/<branch>/<folder>/job-1234567.log")
    return max(logs, key=os.path.getmtime)


def read_log(src):
    """Text of a local log or a GitHub/raw URL, plus the file's mtime (None for URLs)."""
    if re.match(r"https?://", src):
        from urllib.request import urlopen
        url = re.sub(r"^https://github\.com/([^/]+)/([^/]+)/blob/", r"https://raw.githubusercontent.com/\1/\2/", src)
        with urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8", errors="replace"), None
    return open(src, encoding="utf-8", errors="replace").read(), os.path.getmtime(src)


def parse(text):
    s = dict(t0=None, t=None, run_dir=None, bs=None, accum=None, n_train=None, epochs=None, total_steps=None,
             val_every=None, n_val=None, t_train=None, epoch=0, ep_n=0, ep_tot=None, spb=None, step=0,
             budget=None, val_sec=None, val_spb=None, vals=[], stopped=None, evals=[], gen=None, loss_bar=None,
             t_report=None, done=False, error=None, resumed=False)
    day = 0
    prev = None
    for seg in re.split(r"[\r\n]+", text):
        for part in re.split(r"(?=\[\d\d:\d\d:\d\d\] )", seg):
            m = TS.match(part)
            if m:
                sec = hms(":".join(m.groups()))
                if prev is not None and sec < prev - 3600:      # passed midnight
                    day += 86400
                prev = sec
                s["t"] = sec + day
                if s["t0"] is None:
                    s["t0"] = s["t"]
                msg = part[m.end():].strip()
                event(s, msg)
            else:
                b = BAR.search(part)
                if b:
                    progress(s, b)
            if "Traceback (most recent call last)" in part or "CUDA out of memory" in part:
                s["error"] = part.strip()[:160]
    return s


def event(s, msg):
    t = s["t"]
    m = None

    def _m(x):
        nonlocal m
        m = x
        return x

    if msg.startswith("Run dir:"):
        s["run_dir"] = msg.split(":", 1)[1].strip()
    elif _m(re.search(r"batch=(\d+) x accum (\d+)", msg)):
        s["bs"], s["accum"] = int(m[1]), int(m[2])
    elif msg.startswith("Resuming from"):
        s["resumed"] = True
    elif _m(re.match(r"Training: (\d+) images x (\d+) epoch\(s\) = (\d+) optimiser steps \| validating every (\d+) steps on (\d+)", msg)):
        s.update(n_train=int(m[1]), epochs=int(m[2]), total_steps=int(m[3]), val_every=int(m[4]), n_val=int(m[5]), t_train=t)
    elif _m(re.match(r"ep (\d+)/\d+ step (\d+)/\d+ .*budget (\d+)/(\d+)min", msg)):
        s.update(epoch=int(m[1]), step=int(m[2]), budget=(int(m[3]), int(m[4]), t))
    elif _m(re.match(r"VAL step (\d+) .*val ([\d.na]+)", msg)):
        s["step"] = max(s["step"], int(m[1]))
        s["vals"].append((int(m[1]), float(m[2])))
    elif msg.startswith("Early stopping"):
        s["stopped"] = "early stopping (validation loss stopped improving)"
    elif msg.startswith("Time budget reached"):
        s["stopped"] = "training time budget reached"
    elif _m(re.match(r"Evaluating (.+?)(?: on (\d+) test images)? \.\.\.", msg)):
        s["evals"].append(dict(name=m[1], n=int(m[2]) if m[2] else None, t=t))
        s["gen"], s["loss_bar"] = None, None
    elif _m(re.search(r"generated (\d+)/(\d+)", msg)):
        if s["evals"]:
            s["evals"][-1]["n"] = int(m[2])
        s["gen"] = dict(s["gen"] or {}, done=int(m[1]), total=int(m[2]), t=t)
    elif msg.startswith("Saved report files"):
        s["t_report"] = t
    elif msg == "DONE":
        s["done"] = True


def progress(s, b):
    desc, n, tot = b["desc"].strip(), int(b["n"]), int(b["tot"])
    rate = float(b["rate"]) if b["rate"] not in (None, "?") else None
    spb = (rate if b["unit"] == "s/it" else 1 / rate) if rate else None
    m = re.match(r"Training epoch (\d+)/\d+", desc)
    if m:
        s.update(epoch=int(m[1]), ep_n=n, ep_tot=tot)
        if spb and n >= 3:
            s["spb"] = spb
    elif desc.startswith("Validation") and n == tot:
        s["val_sec"] = hms(b["el"])
        s["val_spb"] = s["val_sec"] / max(tot, 1)
    elif desc.startswith("Testing"):
        s["loss_bar"] = dict(n=n, tot=tot, spb=spb or s["val_spb"])
    elif desc.endswith("testing") and spb and n >= 1:
        s["gen"] = dict(s["gen"] or {}, sec_per_batch=spb, batches=(n, tot))


def estimate(s, P, now):
    """Seconds left per phase: dict(setup, training, testing, report) plus labels."""
    bs, accum = s["bs"] or P["per_device_bs"], s["accum"] or P["grad_accum"]
    n_cond = 3 if P.get("eval_no_note", True) else 2
    out = dict(train_frac=0.0, test_frac=0.0, notes=[])
    test_spb = s["val_spb"] or ((s["spb"] or 2.0) * 0.6)
    n_gen = P["n_eval_gen"]
    loss_batches = math.ceil(P["max_test_examples"] / bs)
    gen_cap = P["eval_minutes"] * 60 / n_cond
    sec_img = GEN_SEC_PER_IMAGE
    measured_gen = False
    if s["gen"] and s["gen"].get("sec_per_batch"):
        sec_img, measured_gen = s["gen"]["sec_per_batch"] / P["gen_bs"], True

    def gen_time(n):
        return min(n * sec_img, gen_cap)

    # --- training ---
    if s["t_train"] is None:
        out["training"] = None
    elif s["stopped"] or s["evals"]:
        out["training"], out["train_frac"] = 0, 1.0
    else:
        nb = math.ceil(s["n_train"] / bs)
        ep = max(s["epoch"], 1)
        in_ep = nb - (s["ep_tot"] or nb) + s["ep_n"]               # ep_tot < nb after a resume mid-epoch
        done_b, total_b = (ep - 1) * nb + in_ep, s["epochs"] * nb
        spb = s["spb"] or 2.0
        step_now = done_b / accum
        vals_left = (s["total_steps"] // s["val_every"] - int(step_now // s["val_every"])) + (s["epochs"] - ep + 1)
        val_sec = s["val_sec"] or math.ceil(s["n_val"] / bs) * spb * 0.6
        left = (total_b - done_b) * spb + max(vals_left, 0) * val_sec
        if s["budget"]:
            used, cap, t = s["budget"]
            left = min(left, cap * 60 - used * 60 - (now - t))
        out.update(training=max(left, 0), train_frac=done_b / total_b, epoch=ep, step=int(step_now))
        if not s["spb"]:
            out["notes"].append("training speed not measured yet; estimate firms up after a few batches")

    # --- testing ---
    full_cond = [loss_batches * test_spb + gen_time(n_gen)] * 2 + [gen_time(n_gen)] * (n_cond - 2)
    if s["t_report"] or s["done"]:
        out["testing"], out["test_frac"] = 0, 1.0
    elif not s["evals"]:
        out["testing"] = sum(full_cond)
    else:
        i = len(s["evals"]) - 1
        cur = s["evals"][-1]
        elapsed = now - cur["t"]
        if s["gen"] and s["gen"].get("total"):
            g = s["gen"]
            left_cur = (g["total"] - g["done"]) * sec_img
            left_cur = min(left_cur, max(gen_cap - (now - g["t"]) - (g["done"] * sec_img), 0) + 60) if g["done"] else left_cur
        elif s["loss_bar"] and i < 2:
            lb = s["loss_bar"]
            left_cur = (lb["tot"] - lb["n"]) * (lb["spb"] or test_spb) + gen_time(cur["n"] or n_gen)
        else:
            left_cur = max(full_cond[min(i, n_cond - 1)] - elapsed, 30)
        left = left_cur + sum(full_cond[i + 1:])
        out["testing"] = left
        spent = now - s["evals"][0]["t"]
        out["test_frac"] = spent / max(spent + left, 1)
        out["cond"] = (i + 1, n_cond, cur["name"])
    out["test_measured"] = measured_gen
    out["report"] = 0 if s["done"] else REPORT_SEC
    return out


def render(path, P, now_override=None, walltime_h=24.0, width=36):
    text, mtime = read_log(path)
    s = parse(text)
    if s["t0"] is None:
        return f"{os.path.basename(path)}: nothing logged yet (still loading modules / checking the GPU)."
    if s["run_dir"] and not P.get("_from_file"):
        try:
            P.update(json.load(open(os.path.join(s["run_dir"], "run_config.json"))))
        except OSError:
            pass
    # "now" on the log's clock (hh:mm:ss, same machine clock as the job)
    if now_override:
        now_sod = hms(now_override)
    else:
        n = datetime.now()
        now_sod = n.hour * 3600 + n.minute * 60 + n.second
    ahead = (now_sod - s["t"] % 86400) % 86400
    now = s["t"] + (0 if ahead > 86400 - 3600 else ahead)    # clock a bit behind the last line: not "tomorrow"
    if now_override:
        stale = 0
    elif mtime is None:                           # URL snapshot: age of its last timestamped line
        stale = (now - s["t"]) / 60
    else:
        stale = (time.time() - mtime) / 60
    elapsed = now - s["t0"]
    e = estimate(s, P, now)

    last = s["t"] % 86400
    lines = [f"ED-VLM job progress   ({os.path.basename(path)}, last line {last // 3600:02d}:{last // 60 % 60:02d})"]
    if s["done"]:
        lines += [f"[{bar(1, width)}] 100%   finished in {dur(s['t'] - s['t0'])}",
                  f"Results: {s['run_dir'] or 'the run dir'}  (evaluation_ci.csv, test_predictions.csv)"]
        return "\n".join(lines)
    if s["error"]:
        lines.append(f"!! The log shows an error: {s['error']}")
        lines.append("   The job has probably crashed; look at the end of the log.")
    if stale > 15 and mtime is None:
        lines.append(f"!! This copy of the log ends {dur(stale * 60)} ago; push a fresh copy for an up-to-date estimate.")
    elif stale > 15:
        lines.append(f"!! The log has not changed for {dur(stale * 60)}: the job may have stopped. Check `qstat -u $USER`.")

    if e["training"] is None:
        lines += [f"[{bar(0.02, width)}]   setting up (data, image cache, model load, speed test) · {dur(elapsed)} so far",
                  "    Time-left estimate appears once training starts. Expect roughly 7-8 h in total."]
        return "\n".join(lines)

    left = e["training"] + e["testing"] + e["report"]
    frac = elapsed / max(elapsed + left, 1)
    finish = datetime.now() + timedelta(seconds=left)
    lines.append(f"[{bar(frac, width)}] {frac:4.0%}   elapsed {dur(elapsed)} · about {dur(left)} left · done around {finish:%a %H:%M}")
    setup = s["t_train"] - s["t0"]
    lines.append(f"  setup     {bar(1, 10)}  done ({dur(setup)})")
    if e["train_frac"] >= 1:
        why = f" - {s['stopped']}" if s["stopped"] else ""
        lines.append(f"  training  {bar(1, 10)}  done{why}")
    else:
        lines.append(f"  training  {bar(e['train_frac'], 10)}  epoch {e['epoch']}/{s['epochs']} · step ~{e['step']}/{s['total_steps']}"
                     f" · ~{dur(e['training'])} left")
    n_cond = 3 if P.get("eval_no_note", True) else 2
    if "cond" in e:
        i, k, name = e["cond"]
        g = s["gen"]
        gtxt = f" · {g['done']}/{g['total']} images" if g and g.get("total") else ""
        lines.append(f"  testing   {bar(e['test_frac'], 10)}  model {i}/{k} ({name}){gtxt} · ~{dur(e['testing'])} left")
    else:
        rough = "" if e["test_measured"] else " (rough until testing starts)"
        lines.append(f"  testing   {bar(0, 10)}  {n_cond} models x {P['n_eval_gen']} images · ~{dur(e['testing'])}{rough}")
    lines.append(f"  report    {bar(0, 10)}  ~{dur(e['report'])}")
    if s["vals"]:
        best = min(s["vals"], key=lambda x: x[1])
        lines.append(f"Val loss: {s['vals'][0][1]:.3f} -> {s['vals'][-1][1]:.3f}  (best {best[1]:.3f} at step {best[0]}, "
                     f"{len(s['vals'])} checks)")
    if elapsed + left > walltime_h * 3600:
        lines.append(f"Walltime: {dur(elapsed)} of {walltime_h:g}h - will NOT fit. When it stops, qsub the same script again; it resumes.")
    else:
        lines.append(f"Walltime: {dur(elapsed)} of {walltime_h:g}h - fits.")
    if e["train_frac"] < 1:
        lines.append(f"Training stops early if val loss fails to improve {P.get('patience', 3)} checks in a row, so this is an upper estimate.")
    lines += [f"Note: {n}" for n in e["notes"]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="job log (default: newest in pbs_logs)")
    ap.add_argument("--watch", type=int, metavar="SEC", help="refresh every SEC seconds")
    ap.add_argument("--config", help="run_config.json to use instead of the one in the run dir")
    ap.add_argument("--walltime", type=float, default=24.0, metavar="HOURS")
    ap.add_argument("--now", help=argparse.SUPPRESS)          # HH:MM:SS, for replaying an old log
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    P = dict(DEFAULTS)
    if a.config:
        P.update(json.load(open(a.config)), _from_file=True)
    while True:
        out = render(a.log or newest_log(), dict(P), a.now, a.walltime)
        if not a.watch:
            print(out)
            break
        print("\033[2J\033[H" + out + f"\n\n(refreshing every {a.watch}s - Ctrl+C to quit)", flush=True)
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
