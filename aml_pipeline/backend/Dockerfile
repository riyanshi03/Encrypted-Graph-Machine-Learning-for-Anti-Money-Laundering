FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for snapml, concrete-ml, etc.
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    git \
    libomp-dev \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with correct permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R nobody:nogroup /app && \
    chmod -R 777 /app/data /app/models /app/logs

# Copy application code
COPY . .

# Set permissions for copied files
RUN chown -R nobody:nogroup /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=5000 \
    CONCRETE_ML_URL=http://concrete_ml:8000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Health check with curl
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://0.0.0.0:5000/health || exit 1

# Run the application with proper user
USER nobody
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
