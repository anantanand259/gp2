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

## Run and publish

1. Use a working Python installation. The current `.venv\Scripts\python.exe`
   failed to launch in this workspace. From `C:\AI_ML\gp2`, create a fresh
   environment (for example, `py -3.12 -m venv .venv-rag`) and run
   `.\.venv-rag\Scripts\python.exe -m pip install -r requirements.txt`.
2. Set `GOOGLE_API_KEY` and `OPENROUTER_API_KEY` in the root `.env` file.
   Gemini access is required for scanned documents, provider fallback and web
   search. Set `NGROK_AUTH_TOKEN` if exposing your local server using ngrok.
3. Start the server with `.\.venv-rag\Scripts\python.exe rag_backend\server.py`.
   Keep your existing `rag_backend\knowledge_base` directory; no reset is needed.
4. For GitHub Pages, expose the server through a reachable HTTPS tunnel.
   Cloudflare cannot reach `localhost` on your computer. Set its current URL with
   `npx wrangler secret put RAG_BACKEND_URL`. Set the Worker's existing
   `OPENROUTER_API_KEY` and `GEMINI_API_KEY` secrets if needed.
5. Run `npm ci`, then `npm run deploy` to publish the Worker changes.
   Publish the modified static files through your normal GitHub Pages deployment.
   These code changes have not been deployed by this task.
6. In Admin, set the backend URL to the Worker URL (or directly to the HTTPS
   Python tunnel) and use **Test Connection**. It must report a ready retriever.

Browser-local entries from earlier failed saves are not automatically in the
server database. Open each affected text entry and save it again. Uploads that
previously failed extraction should be retried. Admin file cards remain a local
browser view; the backend database is the source used by the chatbot.
The existing Delete/Clear buttons remove that browser view only; they do not
delete server documents. This fix does not implement backend deletion.

## Validation

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
