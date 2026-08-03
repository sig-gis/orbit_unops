# ---- Base image -------------------------------------------------------
FROM python:3.10-slim

# ---- OS-level hygiene -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Python dependencies ----------------------------------------------
# Copy only the requirements file first so Docker can cache this layer
# separately from the source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application source -----------------------------------------------
COPY . .

# ---- Runtime ----------------------------------------------------------
# Uvicorn binds to 0.0.0.0 so the container port is reachable from the
# host. The module path matches the package layout:
#   orbit_unops/pipeline/api.py  ->  orbit_unops.pipeline.api:app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn pipeline.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
