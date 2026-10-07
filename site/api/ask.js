// Vercel / Netlify-style serverless function: POST /api/ask  {question, history?, page?, case?}
// -> 200 {answer, provider, model} | 4xx/5xx {error}. The page falls back to its offline FAQ on any error.
import { answer } from "./_ask_core.js";

const hits = new Map(); // best-effort per-instance rate limit: 20 requests / 5 min / IP
function limited(ip) {
  const now = Date.now(), win = 5 * 60 * 1000;
  const arr = (hits.get(ip) || []).filter(t => now - t < win);
  arr.push(now); hits.set(ip, arr);
  return arr.length > 20;
}

export default async function handler(req, res) {
  const allowed = process.env.ALLOWED_ORIGIN || "";        // e.g. https://qure-lab.vercel.app ; empty = same-origin only
  const origin = req.headers.origin || "";
  if (allowed && origin === allowed) { res.setHeader("access-control-allow-origin", allowed); res.setHeader("vary", "origin"); }
  if (req.method === "OPTIONS") {
    res.setHeader("access-control-allow-methods", "POST"); res.setHeader("access-control-allow-headers", "content-type");
    return res.status(204).end();
  }
  if (req.method === "GET") return res.status(200).json({ ok: true, providers: { gemini: !!process.env.GEMINI_API_KEY, grok: !!process.env.XAI_API_KEY } });
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const ip = String(req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "?").split(",")[0].trim();
  if (limited(ip)) return res.status(429).json({ error: "Too many questions. Wait a few minutes." });
  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const out = await answer({ question: body.question, history: body.history, page: body.page, caseCtx: body.case }, process.env);
    res.setHeader("cache-control", "no-store");
    return res.status(200).json(out);
  } catch (e) {
    // never echo provider error bodies or keys to the client
    console.error("ask failed:", e.message, e.detail || "");
    return res.status(e.status || 500).json({ error: e.status === 400 ? "Empty question." : "The AI service is unavailable right now." });
  }
}
