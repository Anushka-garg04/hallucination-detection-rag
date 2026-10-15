"""Publish the demo to a public URL - no account needed.

Usage:  python publish.py      (or double-click publish.bat)

What it does:
  1. Starts the Streamlit app locally (if not already running) on port 8501
  2. Downloads cloudflared once (55 MB) if missing
  3. Opens a free Cloudflare quick tunnel and prints a public https URL
"""
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8501
LOCAL_URL = "http://localhost:%d" % PORT
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CF_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ROOT), "Temp")
CF = os.path.join(CF_DIR, "cloudflared.exe")
LOG = os.path.join(ROOT, "tunnel.log")
DETACH = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP


def port_open():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def ensure_streamlit():
    if port_open():
        print("[1/3] Streamlit already running at", LOCAL_URL)
        return
    print("[1/3] Starting Streamlit on port %d ..." % PORT)
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py"),
         "--server.headless", "true", "--server.port", str(PORT)],
        cwd=ROOT, creationflags=DETACH,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        time.sleep(1)
        if port_open():
            print("      Streamlit is up.")
            return
    sys.exit("ERROR: Streamlit failed to start. Run manually: streamlit run app.py")


def ensure_cloudflared():
    if os.path.exists(CF):
        return
    print("[2/3] Downloading cloudflared (~55 MB, one time) ...")
    with urllib.request.urlopen(CF_URL, timeout=120) as r, open(CF, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)


def start_tunnel():
    if os.path.exists(LOG):
        try:
            os.remove(LOG)
        except OSError:
            pass
    print("[3/3] Opening public tunnel ...")
    log = open(LOG, "w", encoding="utf-8", errors="ignore")
    try:
        subprocess.Popen(
            [CF, "tunnel", "--url", LOCAL_URL, "--no-autoupdate"],
            cwd=ROOT, creationflags=DETACH, stdout=subprocess.DEVNULL, stderr=log,
        )
    finally:
        log.close()
    for _ in range(40):
        time.sleep(1)
        try:
            with open(LOG, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
        if urls:
            url = urls[-1]
            print("")
            print("  PUBLIC URL ->", url)
            print("")
            print("  Open it in any browser (works from any device).")
            print("  A NEW url is generated every time you restart - copy it each time.")
            print("  Keep this window open; closing it stops the public link.")
            return url
    sys.exit("ERROR: tunnel did not start - check tunnel.log")


if __name__ == "__main__":
    ensure_streamlit()
    ensure_cloudflared()
    start_tunnel()
