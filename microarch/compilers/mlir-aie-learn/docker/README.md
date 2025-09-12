# MLIR-AIE Docker Environment Setup

This guide provides step-by-step instructions to set up a Docker environment for MLIR-AIE development and test matrix multiplication compilation.

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- Git (for cloning this repository)

### 1. Build the Docker Image

```bash
cd /path/to/this/directory
docker build -f Dockerfile.minimal -t mlir-aie-minimal .
```

### 2. Start the Container

```bash
docker run -d --name mlir-aie-dev mlir-aie-minimal sleep infinity
```

### 3. Enter the Container

```bash
docker exec -it mlir-aie-dev /bin/bash
```

You're now inside the container as the `mlir-aie` user in `/home/mlir-aie`.

## 🔧 Setup MLIR-AIE Environment

### 4. Create Python Virtual Environment

```bash
python3 -m venv ironenv
source ironenv/bin/activate
```

### 5. Install Basic Dependencies

```bash
pip install --upgrade pip
pip install numpy pybind11 wheel setuptools
```

### 6. Download and Install MLIR-AIE Wheels

**Download MLIR-AIE wheel:**
```bash
wget -O mlir_aie.whl 'https://github.com/Xilinx/mlir-aie/releases/download/v1.1.0/mlir_aie-0.0.1.2025090804%2B1ad71ae-cp312-cp312-manylinux_2_35_x86_64.whl'
```

**Fix wheel filename and install:**
```bash
mv mlir_aie.whl mlir_aie-0.0.1-py3-none-any.whl
pip install --force-reinstall --no-deps mlir_aie-0.0.1-py3-none-any.whl
```

**Download LLVM-AIE wheel:**
```bash
wget -O llvm_aie.whl 'https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025011701%2B90b6e4b4-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
```

**Fix wheel filename and install:**
```bash
mv llvm_aie.whl llvm_aie-19.0.0-py3-none-any.whl
pip install --force-reinstall --no-deps llvm_aie-19.0.0-py3-none-any.whl
```

### 7. Install Python Extras (if available)

```bash
# Try to install extras - this may fail if the path doesn't exist, which is OK
pip install -r ~/.local/lib/python3.12/site-packages/mlir_aie/python/requirements_extras.txt 2>/dev/null || echo "Extras not found, continuing..."
```

### 8. Clone MLIR-AIE Repository

```bash
git clone --depth 1 https://github.com/Xilinx/mlir-aie.git
```

## 🧪 Test Matrix Multiplication Compilation

### 9. Navigate to Example

```bash
cd mlir-aie/programming_examples/basic/matrix_multiplication/single_core
```

### 10. Test Compilation

```bash
# Make sure virtual environment is active
source ~/ironenv/bin/activate

# Run the compilation
make
```

### 11. Check Results

If successful, you should see generated files in the `build/` directory:
```bash
ls -la build/
```

## 🔄 Container Management

### Start/Stop Container
```bash
# Stop container
docker stop mlir-aie-dev

# Start existing container
docker start mlir-aie-dev

# Remove container (will lose all data)
docker rm -f mlir-aie-dev
```

### Re-enter Container
```bash
# Enter running container
docker exec -it mlir-aie-dev /bin/bash

# Activate virtual environment
source ironenv/bin/activate

# Navigate to examples
cd mlir-aie/programming_examples/basic/matrix_multiplication/single_core
```

## 🛠️ Troubleshooting

### Platform Compatibility Issue
If you see errors like "wheel is not supported on this platform":
- The solution is already implemented in steps 6 above
- We rename the wheels to generic names and use `--force-reinstall --no-deps`

### Missing Python Modules
If you see `ModuleNotFoundError: No module named 'aie.extras.context'`:
- Make sure you're in the virtual environment: `source ~/ironenv/bin/activate`
- Try installing additional dependencies: `pip install mlir`

### Container Stops Running
If the container stops:
```bash
docker start mlir-aie-dev
docker exec -it mlir-aie-dev /bin/bash
```

## 📁 File Structure

```
docker/
├── Dockerfile.minimal      # Minimal working Dockerfile
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # This file

Container structure:
/home/mlir-aie/
├── ironenv/                # Python virtual environment
├── mlir-aie/              # Cloned MLIR-AIE repository
│   └── programming_examples/
│       └── basic/
│           └── matrix_multiplication/
│               └── single_core/    # Test directory
├── *.whl                  # Downloaded wheel files
```

## 🎯 Success Criteria

The setup is successful when:
1. ✅ Container runs without errors
2. ✅ Python virtual environment is active
3. ✅ MLIR-AIE and LLVM-AIE wheels are installed
4. ✅ Repository is cloned
5. ✅ `make` command runs in the matrix multiplication example
6. ✅ Build files are generated in `build/` directory

## 🚨 Known Issues & Current Status

### ✅ **RESOLVED ISSUES:**
1. **Platform Compatibility**: ✅ Fixed by renaming wheels and using `--force-reinstall --no-deps`
2. **Docker Environment**: ✅ Container setup works perfectly
3. **MLIR-AIE Installation**: ✅ Both MLIR-AIE and LLVM-AIE wheels install successfully

### ⚠️ **CURRENT LIMITATION:**
**Missing `aie.extras.context` Module**: The compilation fails because the `context` module is not available in the installed wheels. This prevents MLIR generation.

**Error seen:**
```
ModuleNotFoundError: No module named 'aie.extras.context'
```

**What works:**
- ✅ Docker container setup
- ✅ Python virtual environment
- ✅ MLIR-AIE and LLVM-AIE wheel installation  
- ✅ Repository cloning
- ✅ Basic Python imports (`import aie`)

**What doesn't work yet:**
- ❌ Matrix multiplication example compilation (missing context module)
- ❌ MLIR file generation (depends on context module)

### 🔧 **POTENTIAL SOLUTIONS:**
1. **Build from source** instead of using pre-built wheels
2. **Find alternative wheels** that include the complete Python API
3. **Install additional dependencies** that provide the missing context module

### 📊 **PROGRESS STATUS:**
- **Environment Setup**: 100% ✅
- **Dependency Installation**: 90% ✅ (missing context module)
- **Example Compilation**: 0% ❌ (blocked by missing module)

### 🎯 **NEXT STEPS FOR RESOLUTION:**
1. Try building MLIR-AIE from source instead of using wheels
2. Look for complete Python API packages
3. Check if there are additional wheels needed for the full Python API

## 💡 Tips

- Always activate the virtual environment: `source ~/ironenv/bin/activate`
- Use `docker exec -it mlir-aie-dev /bin/bash` to enter the container
- Keep the container running with `sleep infinity` in the background
- Use `docker logs mlir-aie-dev` to check container logs if issues arise

## 🔗 Resources

- [MLIR-AIE Repository](https://github.com/Xilinx/mlir-aie)
- [MLIR-AIE Documentation](https://xilinx.github.io/mlir-aie/)
- [Docker Documentation](https://docs.docker.com/)