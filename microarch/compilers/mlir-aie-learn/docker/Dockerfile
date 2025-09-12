FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    clang-14 \
    lld \
    lld-14 \
    cmake \
    ninja-build \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    gcc-13 \
    g++-13 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create mlir-aie user
RUN useradd -m -s /bin/bash mlir-aie && \
    usermod -aG sudo mlir-aie && \
    echo "mlir-aie ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to mlir-aie user
USER mlir-aie
WORKDIR /home/mlir-aie

# Create virtual environment
RUN python3 -m venv ironenv

# Activate virtual environment and upgrade pip
RUN . ironenv/bin/activate && \
    python3 -m pip install --upgrade pip

# Install common Python dependencies first
RUN . ironenv/bin/activate && \
    python3 -m pip install numpy pybind11 wheel setuptools

# Clone mlir-aie repository for examples and headers
RUN git clone --depth 1 https://github.com/Xilinx/mlir-aie.git

# Set up environment
RUN echo 'source ~/ironenv/bin/activate' >> ~/.bashrc

# Create a setup script for easy environment activation
RUN echo '#!/bin/bash' > setup_env.sh && \
    echo 'source ~/ironenv/bin/activate' >> setup_env.sh && \
    echo 'export PYTHONPATH=/home/mlir-aie/mlir-aie/python:$PYTHONPATH' >> setup_env.sh && \
    echo 'echo "Environment set up successfully!"' >> setup_env.sh && \
    echo 'echo "PYTHONPATH: $PYTHONPATH"' >> setup_env.sh && \
    echo 'echo "Current directory: $(pwd)"' >> setup_env.sh && \
    chmod +x setup_env.sh

# Set working directory to programming examples
WORKDIR /home/mlir-aie/mlir-aie/programming_examples

# Default command
CMD ["/bin/bash"]