# Render deployment (free plan)

The root `render.yaml` deploys `rag_backend/server.py` via Gunicorn with one
worker and four threads. The build downloads the ONNX embedding model without
importing the server or accessing your local knowledge base. Secrets and private
documents are not in Git. The lightweight requirements exclude unused PyTorch
and sentence-transformers dependencies.

## Deploy

1. Sign in at https://dashboard.render.com/ and select **New > Blueprint**.
2. Connect https://github.com/anantanand259/gp2, branch **main**. Use the
   root `render.yaml`; review that the service plan is **Free**.
3. Enter `GOOGLE_API_KEY` and `OPENROUTER_API_KEY` in Render's secret fields.
   Use your real keys from your local `.env`, never in Git or chat. If using only
   Gemini, use `YOUR_OPENROUTER_KEY_HERE` as the unused placeholder; it is skipped.
   Add `GROQ_API_KEY` in the service's Environment settings if you have one.
   Do not set `NGROK_AUTH_TOKEN`; Gunicorn does not launch ngrok.
4. Deploy. The first build installs dependencies and caches the embedding model.
   Confirm the service is **Live** and `https://YOUR-SERVICE.onrender.com/api/health`
   returns `status: ok` and `retriever_ready: true`.
5. Point the existing Cloudflare Worker at the actual Render service URL:

   ```powershell
   npx wrangler secret put RAG_BACKEND_URL
   ```

   Paste the HTTPS service URL with no `/api` suffix. This replaces the ngrok
   destination; the existing GitHub Pages chatbot and admin can keep using
   `https://gp2.anantanand259.workers.dev`.
6. Upload a non-sensitive test notice through Admin, then ask about a unique
   detail. Verify `source_type: rag` in the query response and the correct detail.
   Current documents on your PC are not automatically transferred to Render.

For manual **New > Web Service** setup, leave **Root Directory blank**, use
Python, the free plan, and the build/start commands and env vars from `render.yaml`.
The requirements file is at repository root, not inside `rag_backend`.

The start command is `cd rag_backend && gunicorn --config gunicorn.conf.py server:app`.
Changing directory in the shell before launching Gunicorn ensures the config
file is found. Gunicorn's `--chdir` option alone does not resolve a config path
relative to that destination directory. If an existing Blueprint has the old
command, use **Manual sync** to apply the updated `render.yaml` before deploying.

## Free-plan limits

Render free services sleep when idle and use an ephemeral filesystem. Uploaded
files, Chroma and BM25 data disappear on restarts/redeploys (including idle spin
down). This is a demo setup, not durable production storage. A cold start can
exceed the frontend/proxy timeout; wait until health is ready and retry.
512 MB memory is also a limit: check Render logs for OOM errors before deciding
the free plan handles your document volume and concurrency.

To retain notices without paying Render, the backend needs an external durable
database/object store; simply setting an environment variable does not add one.
That integration is not implemented here. Alternatively use a paid Render disk
mounted at `/var/data` and set `RAG_STORAGE_DIR=/var/data`. The application stores
all KB files beneath that directory. Keep one instance/worker with this local
storage design; provider quotas still apply on every hosting platform.

References: https://render.com/docs/deploy-flask,
https://render.com/docs/free, https://render.com/docs/disks.
