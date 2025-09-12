# Start with the ROCm MLIR image
FROM rocm/mlir:latest

# Add repository for GCC-12
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    g++-12 \
    clang \
    lld \
    libstdc++-12-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update certificates
RUN update-ca-certificates

# Set up Python environment
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy pandas pytest cmake

# Clone mlir-aie repository
WORKDIR /opt
RUN git clone https://github.com/Xilinx/mlir-aie.git
WORKDIR /opt/mlir-aie

# Create and activate virtual environment
RUN python3 -m pip install virtualenv
RUN python3 -m virtualenv ironenv
ENV PATH="/opt/mlir-aie/ironenv/bin:${PATH}"

# Install MLIR-AIE requirements - with SSL verification disabled
RUN . ironenv/bin/activate && \
    python3 -m pip install --trusted-host github.com --trusted-host release-assets.githubusercontent.com \
    mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2 && \
    python3 -m pip install --trusted-host github.com --trusted-host release-assets.githubusercontent.com \
    llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly && \
    HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt && \
    python3 -m pip install -r python/requirements_dev.txt && \
    pre-commit install || true
# # Set environment variables
# ENV PYTHONPATH="/opt/mlir-aie/python:${PYTHONPATH:-}"

# Create a directory for working with examples
WORKDIR /workspace

# Add entrypoint script to activate environment
RUN echo '#!/bin/bash\nsource /opt/mlir-aie/ironenv/bin/activate\nexec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]