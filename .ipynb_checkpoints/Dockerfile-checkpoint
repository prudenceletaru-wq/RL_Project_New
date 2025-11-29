# 1. Base image
FROM python:3.12-slim

# 2. Setting working directory
WORKDIR /app

# 3. Installing system dependencies (git, curl, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Copying requirements.txt first to leverage caching
COPY requirements.txt /app/

# 5. Installing CPU-only PyTorch first (big package, long timeout)
RUN pip install --no-cache-dir --default-timeout=500 \
    torch==2.9.1+cpu \
    torchaudio==2.9.1+cpu \
    torchvision==0.24.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 6. Installing the rest of dependencies (no --no-deps)
RUN pip install --no-cache-dir --default-timeout=500 -r requirements.txt

# 7. Copying the rest of the project files (source code)
COPY . /app

# 8. Exposing FastAPI port
EXPOSE 8000

# 9. Running FastAPI with live reload
CMD ["uvicorn", "API.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
