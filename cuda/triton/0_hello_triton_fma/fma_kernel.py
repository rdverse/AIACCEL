#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
import pandas as pd
import matplotlib.pyplot as plt
import time

# Use PyTorch's device management
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@triton.jit
def compute_kernel(
    x_ptr,  # pointer to first input
    y_ptr,  # pointer to second input
    output_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements each program should process
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Create a range of elements this program will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the last block which might be smaller
    mask = offsets < n_elements
    
    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Simple FMA operation: output = x * y + x
    output = x * y + x
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def compute(x, y):
    n_elements = x.shape[0]
    output = torch.empty_like(x)
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch the kernel
    compute_kernel[(n_blocks,)](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x*y + x, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compute(x, y), quantiles=quantiles)
    
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Test the kernel
if __name__ == "__main__":
    # Create larger test data
    size = 2048*2048  # 4M elements
    BLOCK_SIZE = 16 # higher block sizes - triton has better performance
    
    print(f"Testing with {size} elements")
    print(f"Using block size: {BLOCK_SIZE}")
    
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    
    # Warm up
    print("Warming up...")
    for _ in range(100):
        output = compute(x, y)
    
    # Time the computation
    print("Starting computation...")
    start_time = time.time()
    
    # Run multiple iterations
    iterations = 1000
    for i in range(iterations):
        output = compute(x, y)
        if i % 10 == 0:
            print(f"Completed {i}/{iterations} iterations")
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Average time per iteration: {(end_time - start_time)/iterations:.4f} seconds")
    
    # Run the benchmark
    print("\nRunning benchmark...")
    benchmark.run(print_data=True, show_plots=False) 
    plt.savefig('outputs/fma_kernel_benchmark_triton_BLOCK_SIZE_16.png')