#!/usr/bin/env python3  
import numpy as np
import numba
import time

# ------------------------------
# Parameters
# ------------------------------
N = 8192*8  # 2D matrix of size N x N
dtype = np.float32

# ------------------------------
# Initialize data
# ------------------------------
A = np.random.rand(N, N).astype( dtype=dtype)

# ------------------------------
# Numba-accelerated row-major summation
# ------------------------------
@numba.njit(fastmath=True, parallel=True)
def sum_2d_numba_parallel_rowmajor(arr):
    total = 0.0
    for i in numba.prange(arr.shape[0]):
        #for j in range(arr.shape[1]):
        total += np.sum(arr[i,:])
    return total

@numba.njit(fastmath=True, parallel=False)
def sum_2d_numba_rowmajor(arr):
    total = 0.0
    for i in numba.prange(arr.shape[0]):
        #for j in range(arr.shape[1]):
        total += np.sum(arr[i,:])
    return total

# ------------------------------
# NumPy row-major summation
# ------------------------------
def sum_2d_numpy_rowmajor(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        #for j in range(arr.shape[1]):
        total += np.sum(arr[i,:])
    return total

# ------------------------------
# Run and Time Numba
# ------------------------------
sum_2d_numba_rowmajor(A)  # warm-up
start = time.perf_counter()
total_numba = sum_2d_numba_rowmajor(A)
elapsed_numba = time.perf_counter() - start

# ------------------------------
# Run and Time Numba
# ------------------------------
sum_2d_numba_parallel_rowmajor(A)  # warm-up
start = time.perf_counter()
total_numba_parallel = sum_2d_numba_parallel_rowmajor(A)
elapsed_numba_parallel = time.perf_counter() - start

# ------------------------------
# Run and Time NumPy
# ------------------------------
sum_2d_numpy_rowmajor(A)
start = time.perf_counter()
total_numpy = sum_2d_numpy_rowmajor(A)
elapsed_numpy = time.perf_counter() - start

# ------------------------------
# Report bandwidth and comparison
# ------------------------------
bytes_accessed = A.nbytes
bandwidth_numba = bytes_accessed / elapsed_numba / 1e9  # GB/s
bandwidth_numba_parallel = bytes_accessed / elapsed_numba_parallel / 1e9  # GB/s
bandwidth_numpy = bytes_accessed / elapsed_numpy / 1e9  # GB/s

print(f"Matrix size: {N} x {N} ({A.nbytes / 1e9:.2f} GB)")
print("\n[Numba] Row-major traversal")
print(f"  Sum = {total_numba:.1f}")
print(f"  Time = {elapsed_numba:.6f} s")
print(f"  Bandwidth = {bandwidth_numba:.2f} GB/s")

print("\n[Numba] Row-major traversal parallel")
print(f"  Sum = {total_numba_parallel:.1f}")
print(f"  Time = {elapsed_numba_parallel:.6f} s")
print(f"  Bandwidth = {bandwidth_numba_parallel:.2f} GB/s")

print("\n[NumPy] Row-major traversal")
print(f"  Sum = {total_numpy:.1f}")
print(f"  Time = {elapsed_numpy:.6f} s")
print(f"  Bandwidth = {bandwidth_numpy:.2f} GB/s")