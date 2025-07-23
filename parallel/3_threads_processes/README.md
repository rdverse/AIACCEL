## Threads vs Processes

### Memory Organization
- **Threads**:
  - Each thread has its own stack
  - Share heap memory
  - Share global/static variables
  - Share code segment
  - Share file descriptors

- **Processes**:
  - Each process has its own:
    - Stack
    - Heap
    - Global/static variables
    - Code segment
    - File descriptors
  - No direct memory sharing between processes

### Key Differences
- **Threads**: 
  - Share memory space (variables, heap)
  - Same process ID
  - Need synchronization (mutex) for shared data
  - Lighter weight, faster communication

- **Processes**: 
  - Separate memory space
  - Different process IDs
  - No direct memory sharing
  - More isolated, slower communication

### Examples
1. **Threads Example** (`threads.cc`):
   - Creates 2 threads sharing variable `x`
   - Both threads see same `x` value
   - Run: `g++ threads.cc -pthread && ./a.out`

2. **Processes Example** (`processes.cc`):
   - Creates child process using fork()
   - Each process has its own copy of `x`
   - Run: `g++ processes.cc && ./a.out`

### When to Use
- Use **Threads** for:
  - Tasks that need to share memory
  - I/O bound operations
  - UI responsiveness
  - When you need fast communication between tasks

- Use **Processes** for:
  - CPU-bound tasks
  - When isolation is needed
  - When one process failure shouldn't affect others
  - When you need separate memory spaces