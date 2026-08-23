FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PARCELPILOT_BUILD_DIR=/tmp/parcelpilot

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY evals ./evals

# Parse the PDFs and verify every rule anchor at build time. If the document
# pack and the rules registry have drifted apart, the image fails to build
# rather than shipping a system that answers from a stale number.
RUN python -c "from app.knowledge.rules import get_rules; \
    r = get_rules(); \
    assert not r.verify(), r.verify(); \
    print('rules verified against document pack')"

# Hosts that inject $PORT (Render, Fly, Cloud Run) are honoured; 7860 is the
# Hugging Face Spaces default; 8000 locally.
ENV PORT=8000
EXPOSE 8000 7860

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
