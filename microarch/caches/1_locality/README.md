# Cache locality

- Explores cache locality and memory bandwidth using row-major vs column-major matrix access patterns.

```bash
chmod +x 1_row_col_major.py
./1_row_col_major.py
```
(requires sudo access for printing cache/memory info)

- How hardware cache and memory architecture impact performance?

- Includes scripts for benchmarking with NumPy and Numba, and for querying system cache and memory info.

```bash
chmod +x numba_test.py
./numba_test.py
```

- Useful for understanding microarchitectural effects on data access and computational throughput.