FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py .
COPY config.yaml .
# Ensure we have the sample data available
COPY sample_phishing_data.csv .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/insights logs

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the API
CMD ["uvicorn", "api_backend:app", "--host", "0.0.0.0", "--port", "8000"]
