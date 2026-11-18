"""Headless Chrome screenshots of the main app's case page, optionally after clicking a tab (Files)."""
import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websocket

CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
PORT = 9334
W = 1360


class Browser:
    def __init__(self, profile):
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                                      f"--user-data-dir={Path(profile).resolve()}", "--no-first-run", "--disable-gpu",
                                      "--hide-scrollbars", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                page = next(t for t in requests.get(f"http://127.0.0.1:{PORT}/json/list", timeout=2).json()
                            if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.3)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        self.n = 0
        self.view(900)

    def send(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def view(self, h):
        self.send("Emulation.setDeviceMetricsOverride", width=W, height=h, deviceScaleFactor=1, mobile=False)

    def case(self, url, pid):
        self.send("Page.navigate", url=f"{url}/#/case/{pid}")
        time.sleep(0.5)
        self.send("Page.reload")
        for _ in range(60):                                  # wait until the case header has rendered
            time.sleep(0.5)
            if self.js("document.body.innerText.includes('Case file')"):
                break
        time.sleep(1.5)

    def click_tab(self, starts):
        ok = self.js(f"""(() => {{ const b = [...document.querySelectorAll('button')]
            .find(x => x.innerText.trim().startsWith({json.dumps(starts)})); if (b) b.click(); return !!b; }})()""")
        time.sleep(2.5)                                       # pictures load
        self.js("Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(r => {i.onload = i.onerror = r})))")
        return ok

    def shot(self, path):
        h = min(4000, int(self.js("document.documentElement.scrollHeight") or 900))
        self.view(h)
        time.sleep(0.6)
        data = self.send("Page.captureScreenshot", format="png")["data"]
        Path(path).write_bytes(base64.b64decode(data))
        self.view(900)
        return Path(path).name

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
