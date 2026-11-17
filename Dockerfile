# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy data files
COPY products.json categories.json ./

# Copy all Python scripts
COPY *.py ./

# Create directories for output
RUN mkdir -p visualizations book_histograms backups

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Run as a non-root user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Default command - run the main analysis
CMD ["python", "fixed_model.py"]