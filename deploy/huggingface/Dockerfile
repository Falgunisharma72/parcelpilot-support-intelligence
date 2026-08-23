# Hugging Face Spaces build. Identical to the root Dockerfile except that it
# pins the port Spaces expects and writes its build artefacts somewhere the
# Spaces runtime can write to.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PARCELPILOT_BUILD_DIR=/tmp/parcelpilot \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY evals ./evals

RUN python -c "from app.knowledge.rules import get_rules; \
    r = get_rules(); assert not r.verify(), r.verify(); print('rules verified')"

EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
