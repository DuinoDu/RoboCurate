#!/usr/bin/env python3
"""One-click local launcher; reuses only a Studio server for this workspace."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import webbrowser

APP=Path(__file__).resolve().parent
URL='http://127.0.0.1:8421'
workspace=APP/'workspace';workspace.mkdir(exist_ok=True)


def ready():
    try:
        with urllib.request.urlopen(URL+'/api/health',timeout=3) as r:
            return json.load(r).get('workspace')==str(workspace)
    except Exception:return False


if not ready():
    log=(workspace/'server.log').open('ab')
    p=subprocess.Popen([str(APP/'start.sh')],cwd=APP,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (workspace/'server.pid').write_text(str(p.pid))
    for _ in range(60):
        if ready():break
        if p.poll() is not None:raise SystemExit('启动失败，请查看 '+str(workspace/'server.log'))
        time.sleep(.5)
    else:raise SystemExit('启动超时，请查看 '+str(workspace/'server.log'))
webbrowser.open(URL)
