# Optimized multi-stage Dockerfile for faster builds
# docker build -t torch-floating-point-test .
# docker run --rm torch-floating-point-test python -m pytest test/round.py test/data_types.py -v

# Base stage with PyTorch and system dependencies
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime as base

# Set environment variables for faster builds
ENV PYTHONUNBUFFERED=1
ENV CFLAGS="-O2 -march=native"
ENV CXXFLAGS="-O2 -march=native"
ENV MAX_JOBS=4
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (rarely change - good for caching)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python build dependencies (change less frequently)
RUN pip install --no-cache-dir pytest parameterized numpy ninja

# Set the library path for the extension
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build stage - copy dependency files first for better caching
FROM base as builder
WORKDIR /app

# Copy dependency files first (change less frequently)
COPY pyproject.toml setup.py ./

# Copy source code (changes most frequently)
COPY floating_point/ ./floating_point/
COPY test/ ./test/

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Force rebuild of the extension to ensure compatibility
RUN python setup.py clean --all
RUN python setup.py build_ext --inplace

# Runtime stage - minimal image with just the built extension
FROM base as runtime
WORKDIR /app

# Copy only the built extension and necessary files
COPY --from=builder /app/floating_point/ ./floating_point/
COPY --from=builder /app/build/ ./build/
COPY --from=builder /app/test/ ./test/

# Default command - just provide a shell for manual execution
CMD ["/bin/bash"]
