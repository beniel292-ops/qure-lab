"""Assemble a deployable static site (+ AI proxy function) in site/.

    python tools/build_report.py && python tools/build_web_bundle.py && python tools/build_site.py
    cd site && node dev_server.js            # local test at http://localhost:8787
    # deploy: import site/ into Vercel (or `vercel deploy site`), set env vars GEMINI_API_KEY and/or XAI_API_KEY

Layout:
  site/index.html  qure_bundle.json  QURE_Lab_Research_Report.pdf  qure_lab_standalone.html
  site/api/ask.js  api/_ask_core.js  api/_kb.js        (serverless function; keys only in env vars)
  site/package.json  dev_server.js  test_ask.js  DEPLOY.md
  site/vr/  (copied from vr/dist if it exists)
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")

def main():
    if os.path.exists(SITE):
        shutil.rmtree(SITE)
    os.makedirs(os.path.join(SITE, "api"))
    for f in ("index.html", "qure_bundle.json", "QURE_Lab_Research_Report.pdf", "qure_lab_standalone.html"):
        src = os.path.join(ROOT, "web", f)
        if os.path.exists(src):
            shutil.copy2(src, SITE)
    for f in ("ask.js", "_ask_core.js", "_kb.js"):
        shutil.copy2(os.path.join(ROOT, "server", "api", f), os.path.join(SITE, "api", f))
    for f in ("package.json", "dev_server.js", "test_ask.js", "DEPLOY.md"):
        shutil.copy2(os.path.join(ROOT, "server", f), SITE)
    vr = os.path.join(ROOT, "vr", "dist")
    if os.path.isdir(vr):
        shutil.copytree(vr, os.path.join(SITE, "vr"))
    print("site ready:", sorted(os.listdir(SITE)))

if __name__ == "__main__":
    main()
