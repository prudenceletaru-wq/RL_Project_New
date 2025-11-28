# 1. Base image
FROM python:3.12-slim

# 2. Install system dependencies (IMPORTANT for SB3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy project files
COPY . /app

# 5. Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.9.1+cpu \
    torchvision==0.24.1+cpu \
    torchaudio==2.9.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 6. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout=360

# 7. Install uvicorn standard extras (includes click)
RUN pip install --no-cache-dir uvicorn[standard]

# 8. Expose port
EXPOSE 8000

# 9. Run FastAPI
CMD ["uvicorn", "API.serve_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
