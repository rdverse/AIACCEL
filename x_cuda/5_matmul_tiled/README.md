# Matrix Multiplication with Tiling

This project implements matrix multiplication using CUDA with a tiled approach to improve memory access patterns and performance.

## What is Tiling?

Tiling is a technique that divides the computation into smaller blocks (tiles) that fit in shared memory. This helps in two ways:

1. **Memory Access Pattern**: 
   - Without tiling: Each thread accesses global memory in a strided pattern
   - With tiling: Threads load data into shared memory in a coalesced pattern, then access it efficiently

2. **Data Reuse**:
   - Without tiling: Each element is loaded from global memory multiple times
   - With tiling: Elements are loaded once into shared memory and reused

## How Tiling Works

1. **Block Structure**:
   - Each block handles a TILE × TILE portion of the output matrix
   - Threads within a block cooperate to load and compute their tile

2. **Memory Access**:
   ```cpp
   // Load tiles into shared memory
   A_tile[ty][tx] = A[row*WIDTH + m*TILE + tx];
   B_tile[ty][tx] = B[(m*TILE + ty)*WIDTH + col];
   ```

3. **Computation**:
   ```cpp
   // Compute partial sum for this tile
   for(int i = 0; i < TILE; i++) {
       sum += A_tile[ty][i] * B_tile[i][tx];
   }
   ```

## Performance Considerations

1. **Tile Size**:
   - Must be known at compile time for shared memory allocation
   - Common sizes: 16×16, 32×32
   - Larger tiles = more data reuse but less parallelism

2. **Shared Memory**:
   - Limited per block (typically 48KB)
   - Each tile requires 2 × TILE × TILE × sizeof(int) bytes

3. **Synchronization**:
   - `__syncthreads()` ensures all threads have loaded data before computation
   - `__syncthreads()` ensures computation is complete before next tile

## Implementation Details

- Uses shared memory for tile storage
- Implements both tiled and non-tiled versions for comparison
- Benchmarks performance using nvbench
- Supports different matrix sizes and tile sizes

## Building and Running

```bash
mkdir build && cd build
cmake ..
make
./5_matmul_tiled
```

## Results

Below are nvbench results for different tile sizes with matrix size 2048×2048:

| WIDTH | TILE_WIDTH | Samples | CPU Time  | Noise | GPU Time  | Noise | Samples | Batch GPU | Shared Mem (KiB) | Block/Grid | Threads/Block |
|-------|------------|---------|-----------|-------|-----------|-------|---------|-----------|-----------------|------------|---------------|
| 2048  | 8×8        | 1152x   | 28.131 ms | 4.72% | 28.092 ms | 4.67% | 1153x   | 27.601 ms | 0.5            | 8×8/256×256| 64           |
| 2048  | 16×16      | 1184x   | 18.982 ms | 5.42% | 18.952 ms | 5.31% | 1185x   | 19.146 ms | 2.0            | 16×16/128×128| 256         |
| 2048  | 32×32      | 704x    | 18.227 ms | 6.13% | 18.197 ms | 5.99% | 705x    | 18.106 ms | 8.0            | 32×32/64×64| 1024        |
| 2048  | 64×64      | 282608x | 13.208 μs | 260.44%| 1.769 μs | 20.85%| 5568011x| 90.757 ns | 32.0           | 64×64/32×32| 4096        |
| 2048  | 78×78      | 284704x | 13.174 μs | 254.00%| 1.756 μs | 21.20%| 5415828x| 96.419 ns | 47.5           | 78×78/26×26| 6084        |
| 2048  | 96×96      | Error   | -         | -     | -         | -     | -       | -         | 72.0           | 96×96/21×21| 9216        |
| 2048  | 128×128    | N/A     | -         | -     | -         | -     | -       | -         | 128.0          | 128×128/16×16| 16384      |

Without tiling from experiment 4, it took 24.88 ms. So nearly ~20% faster with tiling!!!

Notes:
- Matrix size: 2048×2048
- Shared Memory calculation: (TILE × TILE × 4 bytes × 2 tensors) / 1024 = KiB
- Max Shared Memory: 64 KiB/SM, 48 KiB/Block , so anything above 48KiB/block will fail
- Why does 64x64 fail which uses 32Kb? - due to thread limitation 64*64 = 4096, but we could only have 1024 threads per block
- (Max Threads per Block: 1024)
- Grid dimensions calculated as: (2048 + TILE - 1) / TILE
- Current implementation requires tile size to be hardcoded at compile time
- See code comments for details on dynamic tile size implementation

Limitations:
1. Shared Memory: Each block is limited to 48 KiB of shared memory
2. Thread Count: Each block is limited to 1024 threads
3. Both limitations must be considered when choosing tile size
