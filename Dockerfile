FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/gdown_cache \
    && chmod 755 /tmp/gdown_cache

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory with proper permissions
RUN mkdir -p /app/models && chmod 777 /app/models

# Create temp directory with proper permissions
RUN mkdir -p /tmp/model_downloads && chmod 777 /tmp/model_downloads

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]