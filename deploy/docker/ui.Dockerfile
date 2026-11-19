# syntax=docker/dockerfile:1.7
# Streamlit UI. Talks to the API over the internal network (API_URL).

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "streamlit>=1.40,<2" "requests>=2.32,<3"

RUN groupadd --system app && useradd --system --gid app --home /app app
COPY --chown=app:app ui ./ui
COPY --chown=app:app .streamlit ./.streamlit
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
