// Local test server: serves the built site and /api/ask with the same code Vercel runs.
//   cd site && GEMINI_API_KEY=... XAI_API_KEY=... node dev_server.js   -> http://localhost:8787
// Keys come from your shell, or from a private .env file in the project folder (git-ignored). Never put them in the page.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import handler from "./api/ask.js";

const ROOT = fileURLToPath(new URL(".", import.meta.url));

// Load keys from a local .env file (never committed, never copied into site/ builds).
// Looks in this folder, then the project folder one level up. Existing shell variables win.
import { existsSync, readFileSync } from "node:fs";
for (const f of [join(ROOT, ".env"), join(ROOT, "..", ".env")]) {
  if (!existsSync(f)) continue;
  for (const line of readFileSync(f, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (m && !line.trim().startsWith("#") && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
  console.log("loaded settings from", f);
}
const TYPES = { ".html": "text/html; charset=utf-8", ".json": "application/json", ".js": "text/javascript", ".pdf": "application/pdf",
  ".png": "image/png", ".css": "text/css", ".svg": "image/svg+xml", ".csv": "text/csv" };
const PORT = Number(process.env.PORT || 8787);

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/ask") {
    let raw = ""; for await (const ch of req) raw += ch;
    req.body = raw;
    const shim = { statusCode: 200, setHeader: (k, v) => res.setHeader(k, v),
      status(c) { this.statusCode = c; return this; },
      json(o) { res.writeHead(this.statusCode, { "content-type": "application/json" }); res.end(JSON.stringify(o)); },
      end() { res.writeHead(this.statusCode); res.end(); } };
    return handler(req, shim);
  }
  let p = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
  if (p.endsWith("/")) p += "index.html";
  const file = join(ROOT, p);
  if (!file.startsWith(ROOT) || file.includes(`${ROOT}api`)) { res.writeHead(404); return res.end(); }
  try { await stat(file); res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" }); res.end(await readFile(file)); }
  catch { res.writeHead(404); res.end("not found"); }
}).listen(PORT, () => console.log(`QURE Lab on http://localhost:${PORT}  (gemini key: ${!!process.env.GEMINI_API_KEY}, grok key: ${!!process.env.XAI_API_KEY})`));
