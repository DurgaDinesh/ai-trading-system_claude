FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ta-lib, reportlab, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget curl \
    libatlas-base-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data logs reports/backtests data/models

# Default: run the full bot
CMD ["python", "main.py"]

EXPOSE 8000
