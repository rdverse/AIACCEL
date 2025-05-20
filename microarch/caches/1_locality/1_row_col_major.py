#!/usr/bin/env python3
import numpy as np
import time
import sys
import platform
import time
import pprint
import cpuinfo
import subprocess


def get_dimm_info():
    print("DIMM INFO")
    memspec = subprocess.run(['sudo', 'dmidecode', '-t', 'memory'], text=True,capture_output=True).stdout
    
    ndimms=1
    current = {}
    for line in memspec.splitlines():
        line = line.strip()
        if line.startswith("Memory Device"):
            if current:
                ndimms+=1 
                current = {}
        elif line.startswith("Size:"):
            current["Size"] = line.split(":", 1)[1].strip()
        elif line.startswith("Configured Memory Speed:"):
            current["Speed"] = line.split(":", 1)[1].strip()
        elif line.startswith("Locator:"):
            current["Slot"] = line.split(":", 1)[1].strip()
        elif line.startswith("Type:"):
            current["Type"] = line.split(":", 1)[1].strip()
        elif line.startswith("Part Number:"):
            current["Part"] = line.split(":", 1)[1].strip()
        elif line.startswith("Rank"):
            current["Rank"] = line.split(":", 1)[1].strip()
        elif line.startswith("Total Width:"):
            current["Bus_width"] = line.split(":", 1)[1].strip()
            
    current["ndimms"] = ndimms
    
    print()
    pprint.pprint(current, indent=4)
    
    # print theoretical bandwidth 
    # theoretical bandwidth = speed * memory_channels * bus_width / 8bits
    # my system has dual channel so, channels = 2, if you have quad, then set it to 4
    channels=2
    speed = int(''.join(filter(str.isdigit, current["Speed"]))) 
    ndimms=ndimms
    bus_width = int(''.join(filter(str.isdigit, current["Bus_width"])))
    theoretical_bw = (speed*channels*bus_width)/(8*1000)
    print(f"THEORETICAL BANDWIDTH {theoretical_bw} GB/s") # matches closely with wikichips
    #//bus_width = current
     
def get_cache_info():
    """Retrieve basic CPU and cache information if available."""
    print("CACHE INFO")
    info = {"cpu": platform.processor(), "l1": None, "l2": None, "l3": None}
    def normalize(size):
        if not size:
            return None
        size = str(size).upper()
        if size.endswith("KB") or size.endswith("MB"):
            return size
        try:
            size_int = int(size)
            if size_int >= 1024 * 1024:
                return f"{size_int / (1024 * 1024):.1f} MB"
            elif size_int >= 1024:
                return f"{size_int / 1024:.1f} KB"
            else:
                return f"{size_int} B"
        except:
            return str(size)
    if cpuinfo:
        # cpuinfo gives wrong values for AMD processors
        ci = cpuinfo.get_cpu_info()
        info["cpu"]= normalize(ci.get("brand_raw", info["cpu"]))
        info["l1"] = normalize(ci.get("l1_data_cache_size", None))
        info["l2"] = normalize(ci.get("l2_cache_size", None))
        info["l3"] = normalize(ci.get("l3_cache_size", None))
        #from cpuinfo import get_cpu_info
        #info = get_cpu_info()
        #print(info)
    print("warning - latencies are incorrect")
    cachespec = subprocess.run(['sudo', 'dmidecode', '-t', 'cache'])
    print(cachespec.stdout)


def measure_memory_bandwidth(matrix, access_pattern="row"):
    """Measure time and bandwidth of summing the matrix by row or column."""
    N = matrix.shape[0]
    start = time.perf_counter()

    total = 0.0
    if access_pattern == "row":
        for i in range(N):
            total += matrix[i, :].sum()
    elif access_pattern == "col":
        # Transpose matrix
        #matrix = matrix.T
        for j in range(N):
            total += matrix[:, j].sum()
    else:
        raise ValueError("Unknown access pattern")

    elapsed = time.perf_counter() - start
    num_elements = N * N
    bytes_read = num_elements * matrix.itemsize
    throughput = bytes_read / elapsed / (1024 ** 3)  # GB/s

    return elapsed, throughput, total


def run_benchmark(sizes, dtype=np.float32):
    
    for N in sizes:
        print(f"Matrix Size: {N} x {N}")
        A = np.random.rand(N, N).astype(dtype)
        # warmup
        t_row, bw_row, sum_row = measure_memory_bandwidth(A, "row")
        time.sleep(0.5)
        # run
        t_row, bw_row, sum_row = measure_memory_bandwidth(A, "row")
        time.sleep(2)
        # warmup
        t_col, bw_col, sum_col = measure_memory_bandwidth(A, "col")
        time.sleep(0.5)
        # run
        t_col, bw_col, sum_col = measure_memory_bandwidth(A, "col")

        if not np.isclose(sum_row, sum_col, rtol=1e-6):
            print(f"Warning: sums differ (diff={abs(sum_row-sum_col):.6f})")

        print(f"  Row-major sum time:     {t_row:.6f} s, Throughput: {bw_row:.2f} GB/s")
        print(f"  Column-major sum time:  {t_col:.6f} s, Throughput: {bw_col:.2f} GB/s")
        print(f"  Speedup (row vs col):   {t_col / t_row:.2f}x\n")


# measure flops and arithmetic intensity

if __name__ == "__main__":
    get_cache_info()
    get_dimm_info()
    print("\nBenchmarking matrix summation in numpy on single core with cache locality analysis\n")

    matrix_sizes = [2,32,64,256, 1024, 2048, 4096,8192, 8192*2]
    matrix_sizes = [1024*i for i in range(16)]
    for ms in matrix_sizes:
        # while True:
        run_benchmark([ms])
