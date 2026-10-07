# Deploying QURE Lab (site + Ask QURE AI)

This folder is the whole website: the console (`index.html`), its data (`qure_bundle.json`), the research report PDF, and one serverless function (`api/ask.js`) that lets the assistant use Gemini, with Grok as backup.

## How Ask QURE answers

1. **Built-in guide.** About 25 curated questions are answered instantly from the project's own numbers. This needs no internet and no key, so it works even if every AI service is down.
2. **Gemini → Grok.** For anything else, the page calls `/api/ask`. The function tries Gemini first. If Gemini fails, times out (12 s) or has no key, it tries Grok. Swap the order with `QURE_PROVIDER_ORDER=grok,gemini`.
3. **If both fail,** the page shows the closest guide answers and says the AI is not connected.

The AI only sees the project's fact sheet (generated from the bundle), the current page, and the numbers of the selected case. It is told:

- not to invent numbers;
- not to give medical advice;
- never to reveal test labels.

## API keys: never in the page, never in chat, never in git

Set them as environment variables on the host.

| Variable | Needed | Example |
| --- | --- | --- |
| `GEMINI_API_KEY` | one of the two | from Google AI Studio |
| `XAI_API_KEY` | one of the two | from console.x.ai |
| `GEMINI_MODEL` | optional | default `gemini-3.6-flash` |
| `XAI_MODEL` | optional | default `grok-4.7` |
| `QURE_PROVIDER_ORDER` | optional | `gemini,grok` (default) |
| `ALLOWED_ORIGIN` | optional | your site URL, only if the page is hosted on a different domain |

Model names change often. If a call fails with HTTP 404, check the current model list in each provider's docs and update `GEMINI_MODEL` / `XAI_MODEL`. No code change is needed.

## Vercel (recommended)

1. Push this `site/` folder to a GitHub repo, or run `npx vercel deploy` inside it.
2. In Vercel, go to **Project → Settings → Environment Variables** and add `GEMINI_API_KEY` and `XAI_API_KEY`. Then redeploy.
3. Open `https://<your-app>.vercel.app/api/ask` in a browser. It should show `{"ok":true,"providers":{"gemini":true,"grok":true}}`. The key values are never shown.
4. Open the site, click **Ask QURE**, and ask something not in the FAQ, for example "why did noise change case 0001?". The badge under the answer names the provider.

Netlify and Cloudflare also work, but their function wrappers differ slightly. `api/_ask_core.js` is plain JavaScript and can be reused as-is.

## Local test

```bash
cd site
npm test                                                   # 6 tests, no network
GEMINI_API_KEY=... XAI_API_KEY=... node dev_server.js      # http://localhost:8787
```

Without keys, the site still runs and the assistant answers from the built-in guide.

## Safety notes

- The function limits each IP to 20 questions per 5 minutes and caps questions at 600 characters. It forwards only numbers and an id for the selected case, and never returns provider error details.
- Spend limits: set a monthly budget or quota in Google AI Studio and the xAI console.
- The claude.ai published page cannot call this function, because its sandbox blocks other hosts. There, Ask QURE uses Claude through the viewer's own account, after asking them, or falls back to the built-in guide.
