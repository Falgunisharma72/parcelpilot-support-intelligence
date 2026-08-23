# Deploying

The app is a single container. It parses the PDFs and verifies every rule anchor
at **build** time, so a drifted rules registry fails the build rather than
shipping a system that answers from a stale number.

Only one secret is needed: `ANTHROPIC_API_KEY`. Without it the app still starts —
the Signals and Access log views run entirely on the deterministic layer — and
the chat tab reports that the agent is offline.

---

## Render (free, no card)

1. Push this repo to GitHub (done).
2. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
3. Pick the repo. `render.yaml` is detected; the plan is already set to `free`.
4. When prompted for `ANTHROPIC_API_KEY`, paste it. It is marked `sync: false`,
   so it is never committed.
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

Then add `ANTHROPIC_API_KEY` under **Settings → Variables and secrets → New secret**.

## Fly.io

```bash
fly launch --no-deploy --name parcelpilot-support
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

`$PORT` is honoured, so no config change is needed.

## Anywhere else

```bash
docker build -t parcelpilot-support .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... parcelpilot-support
```
