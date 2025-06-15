# CUDA vs Triton: FMA Operation Example

This README compares CUDA and Triton implementations of a simple Fused Multiply-Add (FMA) operation. Both achieve the same goal but with different syntax and approaches.

## Key Differences

### 1. Kernel Definition
```cuda
// CUDA
__global__ void compute_kernel(float* x, float* y, float* output, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        output[idx] = x[idx] * y[idx] + x[idx];  // Simple FMA
    }
}
```

```python
# Triton
@triton.jit
def compute_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y + x  # Simple FMA
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. Kernel Launch
```cuda
// CUDA
int block_size = 256;
int num_blocks = (n_elements + block_size - 1) / block_size;
compute_kernel<<<num_blocks, block_size>>>(x, y, output, n_elements);
```

```python
# Triton
n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
compute_kernel[(n_blocks,)](x, y, output, n_elements, BLOCK_SIZE=256)
```

### 3. Memory Management
```cuda
// CUDA
float *x, *y, *output;
cudaMalloc(&x, n_elements * sizeof(float));
cudaMalloc(&y, n_elements * sizeof(float));
cudaMalloc(&output, n_elements * sizeof(float));
// ... use memory ...
cudaFree(x);
cudaFree(y);
cudaFree(output);
```

```python
# Triton
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty_like(x)
# Memory automatically managed by PyTorch
```

## Key Concepts

1. **Thread Organization**
   - CUDA: Uses `blockIdx`, `threadIdx`, `blockDim` for thread organization
   - Triton: Uses `program_id` and `tl.arange` for similar functionality

2. **Memory Access**
   - CUDA: Direct pointer arithmetic with bounds checking
   - Triton: Uses `tl.load` and `tl.store` with mask for safe memory access

3. **FMA Operation**
   - Both implement the same FMA operation: `output = x * y + x`
   - This is a fundamental GPU operation that both frameworks optimize for

4. **Compilation**
   - CUDA: Requires separate compilation with nvcc
   - Triton: JIT compilation with Python decorator (@triton.jit)

## Advantages

### CUDA
- More control over low-level details
- Direct access to all CUDA features
- Better for highly specialized kernels

### Triton
- Python-like syntax
- Automatic memory management
- Easier integration with PyTorch
- Built-in performance optimizations

## When to Use Which?

- Use CUDA when:
  - You need maximum control over GPU resources
  - Writing highly specialized kernels
  - Working with legacy CUDA code

- Use Triton when:
  - Quick prototyping is needed
  - Working with PyTorch
  - Want simpler syntax
  - Need automatic optimizations

## Performance Considerations

The FMA operation is a fundamental GPU operation that both frameworks optimize for. The performance difference between CUDA and Triton should be minimal for this simple operation. However, tweaking the block size gives Triton an edge over PyTorch.

**Block Size Impact**:
   - Too small (32): More blocks, more scheduling overhead
   - Optimal (64-256): Better memory coalescing, fewer blocks
   - Too large (>1024): Might exceed GPU resources - automatically downscaled to 1024 in most cases