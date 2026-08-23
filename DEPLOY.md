# Deploying

The app is a single container. It parses the PDFs and verifies every rule anchor
at **build** time, so a drifted rules registry fails the build rather than
shipping a system that answers from a stale number.

Only one secret is needed, and every option has a **free tier**: set any one of
`GROQ_API_KEY` (recommended), `GEMINI_API_KEY`, `CEREBRAS_API_KEY`,
`OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY` — or
`ANTHROPIC_API_KEY` if you would rather pay for it. The provider is detected
automatically; `make providers` confirms which one is active and probes that tool
calling works on it.

Without any key the app still starts — the Signals and Access log views run
entirely on the deterministic layer — and the chat tab explains which free key to
add, with links.

---

## Render (free, no card)

1. Push this repo to GitHub (done).
2. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
3. Pick the repo. `render.yaml` is detected; the plan is already set to `free`.
4. Paste one model key when prompted (blank for the rest). All are marked
   `sync: false`, so nothing is committed.
5. First build takes ~4 minutes. Health check is `/api/health`.

Free instances sleep after 15 minutes idle and take ~30 seconds to wake — worth
hitting the URL a minute before a demo.

## Hugging Face Spaces (free)

```bash
hf auth login                       # if the stored token has expired
hf repo create parcelpilot-support --repo-type space --space_sdk docker

git clone https://huggingface.co/spaces/<your-username>/parcelpilot-support /tmp/pp-space
cp -r app data evals requirements.txt /tmp/pp-space/
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md /tmp/pp-space/
cd /tmp/pp-space && git add -A && git commit -m "ParcelPilot Support Intelligence" && git push
```

Then add one model key under **Settings → Variables and secrets → New secret**
(e.g. `GROQ_API_KEY`).

## Fly.io

```bash
fly launch --no-deploy --name parcelpilot-support
fly secrets set GROQ_API_KEY=gsk_...
fly deploy
```

`$PORT` is honoured, so no config change is needed.

## Anywhere else

```bash
docker build -t parcelpilot-support .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... parcelpilot-support
```
