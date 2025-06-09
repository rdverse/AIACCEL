## Setup
```bash
conda env create -f environment.yml
conda activate uarch
'''














## Steps for profiling
1. Install uprof
2. check - sudo cat /proc/sys/kernel/perf_event_paranoid
3. Set - sudo sysctl kernel.perf_event_paranoid=0/2 (0 for system wide and 2 for process)
4. run profiler
5. Set - sudo sysctl kernel.perf_event_paranoid=4


