FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-compile torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir --no-compile -r requirements.txt \
    && python -m pip check

COPY marathon/ marathon/
COPY frontend/ frontend/
COPY models/marathon/ models/marathon/
COPY demo/ demo/
COPY strideplan/ strideplan/
COPY strideplan_pipeline/ strideplan_pipeline/
COPY tests/ tests/

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "marathon.api:app", "--host", "0.0.0.0", "--port", "8000"]
