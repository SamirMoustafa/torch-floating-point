# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY pyproject.toml ./
COPY README.md ./
COPY version.py ./
COPY floating_point/ ./floating_point/
COPY test/ ./test/

# Install only test dependencies (PyTorch is already installed)
RUN pip install pytest parameterized

# Set the library path for the extension (PyTorch image already has correct paths)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install the package in development mode (without building extension)
RUN pip install -e floating_point

# Default command - just provide a shell for manual execution
CMD ["/bin/bash"]
RUN pytest --log-cli-level=DEBUG --capture=tee-sys test/round.py test/data_types.py -v
