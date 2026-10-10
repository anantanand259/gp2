# College chatbot: KB-first fix

Every question now goes through the Python knowledge base. The browser never calls
the general LLM when the backend is unavailable, times out, or refuses an answer.
The Worker `/api/health` checks the Python server rather than reporting proxy liveness.

Hybrid retrieval keeps source metadata. A structured model decision must identify
supporting source indices before a KB answer is accepted. If the retrieved chunks
have no useful answer, the remaining stored chunks are checked in batches before
external search is allowed. Partial KB answers remain KB answers; missing details
are acknowledged. Provider errors and malformed responses return errors, never a
KB miss. The support decision is still an LLM judgment; this is not a mathematical
guarantee of answer correctness. Checking a large KB on a miss costs more model
calls and may exceed the browser's 90-second timeout; that timeout never bypasses KB.

Web fallback uses Gemini's Google Search tool and requires grounding citations.
Ordinary model training knowledge is no longer labeled “internet”. Implementation
follows the [Google Gen AI SDK's search API](https://github.com/googleapis/python-genai#grounding-with-google-search).

Uploads are marked processed only after successful indexing. Stable vector IDs
make retries safe, and the persisted lexical documents repopulate vector entries
on startup. Scanned PDFs use Gemini extraction. Supported files are PDF, JPG,
JPEG, PNG, WEBP, BMP, JSON, TXT, MD and CSV, with a 10 MB request limit.
Already indexed filenames return a conflict instead of silently accepting zero
new chunks; give an updated notice a new filename. Saving manual text waits for
backend indexing. Subsequent edits of entries saved with this version replace
their previous backend chunks using `entry_id`.

The file upload endpoint processes only the selected file. An older unreadable
pending image cannot block a new PDF, JSON or other supported upload. Explicit
folder scans continue past failed files; unreadable selected uploads return
HTTP 422 with `UNREADABLE_DOCUMENT` instead of an unrelated HTTP 500.

Gemini quota failures return HTTP 429 and a safe message with the provider's
retry delay when supplied. Pending uploads are not automatically processed on
server startup; this prevents restarts from consuming more extraction requests.
To explicitly scan a directory, POST to `/api/rag/ingest` without a file, or opt
in to startup scanning with `RAG_INGEST_ON_STARTUP=1`. Images and scanned PDFs
require Gemini visual extraction. JSON, plain text, CSV, Markdown and PDFs with
extractable text are indexed locally. Answer generation may still require
provider quota unless a working OpenRouter key is configured.

## Run and publish

### Automatic model fallback and multiple users

The Python backend tries configured OpenRouter, Groq and Gemini routes in that
order. Gemini includes both `gemini-2.5-flash` and `gemini-2.5-flash-lite` by
default. Keys and model order are configured in `.env`; see `.env.example`.
Missing keys are skipped. Add `GROQ_API_KEY` to activate Groq. Restart Python
after changing keys. A key being accepted does not guarantee model access or
available credits; actual generation can still fail and trigger the next route.

Every fallback receives the same KB context and JSON validation rules. A model
failure never authorizes a general-knowledge answer. Supporting source indices
are kept in API metadata for validation, while the chat answer has no appended
source list, `[Source N]` markers, or source badge.

Quota failures put a route on cooldown using the provider retry delay when
available. Concurrent requests skip cooling-down or busy providers. A semaphore
bounds each provider to `LLM_MAX_CONCURRENCY` calls, default 2. Repeated supported
questions are cached for five minutes, and identical simultaneous questions use
one generation. A changed KB fingerprint invalidates prior cached answers.
State is shared within one Python process; multiple server processes would need
a shared cache/cooldown store. If all configured routes are unavailable, the
backend returns a clear retryable HTTP 503, not an invented answer.

Image/scanned-PDF extraction and grounded web search use the two Gemini models;
Groq and the configured OpenRouter text model do not provide those capabilities.
Different Gemini models may still share project/account limits. Independent
providers help availability but cannot guarantee unlimited traffic or prevent
all outages. The local Python server and public tunnel still need to remain up.
Adapters follow the [Groq chat API](https://console.groq.com/docs/api-reference),
[OpenRouter structured-output API](https://openrouter.ai/docs/guides/features/structured-outputs)
and [Gemini Flash-Lite capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite).

1. From `C:\AI_ML\gp2`, use the existing `.venv\Scripts\python.exe`.
   This environment and its backend dependencies were verified outside the
   Codex sandbox. Install updates with
   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and replace the placeholders with valid
   `GOOGLE_API_KEY` and `OPENROUTER_API_KEY` values. `GEMINI_API_KEY` is also
   accepted as an alias for `GOOGLE_API_KEY` when the latter is unset.
   These are Python backend credentials; Worker secrets are a separate configuration.
   Gemini access is required for scanned documents, provider fallback and web
   search. Set `NGROK_AUTH_TOKEN` if exposing your local server using ngrok.
3. Restart the server with `.\.venv\Scripts\python.exe rag_backend\server.py`
   after changing keys, or use `rag_backend\start_server.bat`.
   Keep your existing `rag_backend\knowledge_base` directory; no reset is needed.
4. For GitHub Pages, expose the server through a reachable HTTPS tunnel.
   Cloudflare cannot reach `localhost` on your computer. Set its current URL with
   `npx wrangler secret put RAG_BACKEND_URL`. Set the Worker's existing
   `OPENROUTER_API_KEY` and `GEMINI_API_KEY` secrets if needed.
5. Run `npm ci`, then `npm run deploy` to publish the Worker changes.
   Publish the modified static files through your normal GitHub Pages deployment.
   Pushing to GitHub does not deploy the Cloudflare Worker or restart Python.
6. In Admin, set the backend URL to the Worker URL (or directly to the HTTPS
   Python tunnel) and use **Test Connection**. It must report a ready retriever.

Browser-local entries from earlier failed saves are not automatically in the
server database. Open each affected text entry and save it again. Uploads that
previously failed extraction should be retried. Admin file cards remain a local
browser view; the backend database is the source used by the chatbot.
The existing Delete/Clear buttons remove that browser view only; they do not
delete server documents. This fix does not implement backend deletion.

## Validation

On diagnosis, the public RAG endpoint returned HTTP 500 with Google's
`API_KEY_INVALID` error. The local backend health reported zero indexed chunks.
A listening backend does not establish that its provider keys work. Update its
keys, restart it and ingest a notice before expecting KB answers. Provider
authentication and quota errors are now returned as safe, actionable errors;
the frontend preserves these instead of saying the backend is unreachable.

Offline regression tests require Python's standard library and Node.js:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/rag-routing.test.cjs
```

They exercise real routing functions with provider stand-ins, frontend behavior,
Worker health forwarding, citation validation and the ingestion failure path.
They do not access private uploaded documents or call paid models.

After starting and publishing, upload a test notice with a distinctive deadline
and ask a paraphrased question about it. Expect `source_type: rag`, the uploaded
deadline and filename citations. Ask an unrelated question and expect cited web
results or a transparent no-answer response. Stop the backend and ask again;
expect a connection error and no general-knowledge answer. Test a scanned PDF and
an unreadable file; only successfully indexed content should receive a success
message. Live extraction, embedding quality and provider output still require
this integration check with working dependencies and API access.
