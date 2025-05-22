## Directory Structure

- `4_matmul.cu` - Basic matrix multiplication implementation
- `4_matmul_nvbench.cu` - NVBench-based benchmarking version
- `CMakeLists.txt` - CMake configuration file for NVBench
- `build.sh` - Build script
- `Makefile` - Alternative make-based build system

## Benchmark Matrix Multiplication

### Manual matmul benchmark 
```bash
# Build and run
make -j$(nproc)
./4_matmul
```

Output: out_matmul.txt

### NVBench matmul benchmark

Option 1 - Using CMake (recommended):
```bash
# Build
rm -rvf build
mkdir build
cd build
cmake ..
make -j$(nproc)

# Run
./4_matmul_nvbench
```

Option 2 - Using build script:
```bash
# Build and run
bash build.sh
./4_matmul_nvbench
```

Output: out_matmul_nvbench.txt 

## Performance Comparison

| Matrix Size | Manual Time (ms) | Manual Noise (%) | NVBench Time (ms) | NVBench Noise (%) |
|------------|------------------|------------------|-------------------|-------------------|
| 1024x1024  | 3.39            | 5.65            | 2.97             | 5.19             |
| 2048x2048  | 24.48           | 5.80            | 24.88            | 4.81             |
| 4096x4096  | 199.53          | 4.07            | 201.12           | 4.49             |

*Note: Both methods follow different approaches. The time reported here is GPU execution time.