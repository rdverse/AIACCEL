## Run command
rm-rvf *.png;rm imageedit_grids_bw;make clean;make all;./imageedit_grids_bw

## Initial setup


```cpp
// --- Pseudocode for Current Timing ---
/*
.
.
cuda even create and start
my_kernel_to_benchmark_1<<<...>>>(...); // Launch the CUDA kernel
event synchronize

cuda even create and start
my_kernel_to_benchmark_2<<<...>>>(...); // Launch the CUDA kernel
event synchronize
.
.
*/

The above implementation had several limitations which were fixed in the current setup

**1. Lack of Pre-Measurement Warm-up**
- The initial kernel invocation includes overhead from driver setup and resource allocation.
- This can skew the first measured timing.
- *Mitigation: Execute a couple warm-up iterations before collecting real timing data.*

**2. Timing Includes Overhead Beyond Kernel Execution**
- Timing currently covers kernel launch delays and setup, not just the compute time.
- This makes it difficult to determine the true processing duration of the kernel.
- *Mitigation: Refine the timing window to focus on actual computation.*

**3. No Mechanism to Clear GPU L2 Cache**
- Performance measurements may be affected by data left in the GPU cache from previous runs.
- This can lead to over-optimistic timing results.
- *Mitigation: Invalidate or randomize memory regions to simulate cache-cold runs.*

**4.1 Fluctuating Measurements Across Runs**
- GPU performance can vary due to frequency scaling, temperature, or background tasks.
- Not accounting for this introduces inconsistency in results.
- *Mitigation: Collect results across multiple runs and analyze their spread.*

**4.2 No Statistical Processing of Timing Data**
- Only a small number of runs are performed, and results are often reported as a single value.
- This approach hides variability and may misrepresent actual performance.
- *Mitigation: Gather a larger sample of measurements, compute statistical summaries, and visualize distributions.*


## Other Points 

- Profilers act as performance debuggers; benchmarks are performance unit tests.
- Reporting a single value is misleading; always use multiple measurements and statistical summaries.
- Embrace measurement noise and use statistics to interpret results.
- lot of boilerplate code for supporting above limitations can be addresed with using nvbench.
- Use formal criteria to determine when enough measurements have been collected.


## Results 

Below is time in ms with about 3-5% run-to-run variation
These results are when the fused kernel is nearly equal to the non-fused kernels implementation i.e., follows exact same steps
In this case, the perf gains for fused kernel were minimal across various filter sizes (see out2.txt)
Method   Filter size   fused   non-fused       
Nsight     100         224     238
Manual     100         227     226 
Nsight     40          40.1    42.8                  
Manual     40          44.7    46.9

The results for slightly more optimized fused kernel is shown in out.txt
- after optimization overall the fused kernel showed better perf

