FROM python:3.11-slim

WORKDIR /app

# Ensure Python stdout/stderr is unbuffered so all startup logs appear in Render dashboard
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=10000

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first (avoids downloading 2.5GB of CUDA/Nvidia wheels that cause OOM on Render 512MB RAM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Render and local development
EXPOSE 10000 8000

# Launch production server with dynamic port support
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
