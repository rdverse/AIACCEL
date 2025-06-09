# Race Conditions in Multi-threaded Programs

This example demonstrates race conditions in multi-threaded programs and their unexpected behaviors.

## The Problem

When two threads try to increment a shared variable simultaneously, we expect the final value to be the sum of all increments. However, due to race conditions, we can see unexpected results:

1. **Values Less Than Expected** (e.g., < 200 for 100 iterations per thread)
2. **Values More Than Expected** (e.g., > 200 for 100 iterations per thread)

## Why This Happens

### 1. Non-Atomic Operations
The increment operation (`pings++`) is not atomic. It consists of three steps:
```assembly
movl    pings(%rip), %eax     # 1. Read from memory
addl    $1, %eax              # 2. Increment in register
movl    %eax, pings(%rip)     # 3. Write back to memory
```

### 2. Race Condition Scenarios

#### Scenario 1: Lost Updates (Values < 200)
```
Thread 1: reads 5
Thread 2: reads 5
Thread 1: increments to 6
Thread 2: increments to 6
Thread 1: writes 6
Thread 2: writes 6
```
Result: Only one increment is counted instead of two.

#### Scenario 2: Cache Line Bouncing (Values > 200)
Due to CPU caching and memory ordering:
- Threads might read old values from their cache
- Writes might be delayed
- Cache might not be immediately synchronized
- CPU might reorder instructions

## Solutions

### 1. Using Mutex
```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* routine(void* arg) {
    for(int i = 0; i < 100; i++) {
        pthread_mutex_lock(&mutex);
        pings++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}
```

### 2. Using Atomic Operations
```c
atomic_int pings = 0;

void* routine(void* arg) {
    for(int i = 0; i < 100; i++) {
        atomic_fetch_add(&pings, 1);
    }
    return NULL;
}
```

## Key Learnings

1. **Atomic Operations**
   - Simple operations like increment are not atomic
   - Need proper synchronization for shared resources
   - Use mutexes or atomic operations for thread safety

2. **Memory Model**
   - CPU caching can affect thread behavior
   - Memory ordering can be different from program order
   - Cache coherency protocols can cause unexpected results

3. **Debugging Race Conditions**
   - Race conditions are non-deterministic
   - Same code can produce different results
   - Need proper synchronization to ensure consistent behavior

4. **Performance Considerations**
   - Mutexes have overhead
   - Atomic operations are more efficient
   - Choose the right synchronization mechanism for your use case

## Building and Running

```bash
gcc -pthread 4_race_cond.c -o race_cond
./race_cond
```

## Expected Output
Without synchronization, you might see:
- Values less than 200 (lost updates)
- Values more than 200 (cache line bouncing)
- Different values on different runs

With proper synchronization, you should always see 200 (100 increments per thread).

## Assembly Code Analysis

### Without Mutex (4_race_cond.s)
```assembly
.L3:
    movl    pings(%rip), %eax     # 1. Load pings from memory to CPU register
    addl    $1, %eax              # 2. Increment value in register
    movl    %eax, pings(%rip)     # 3. Store back to memory
    addl    $1, -4(%rbp)          # Increment loop counter
```
This shows why race conditions occur - the increment operation is not atomic.

### With Mutex (4_race_cond_mutex.s)
```assembly
.L3:
    leaq    mutex(%rip), %rax
    movq    %rax, %rdi
    call    pthread_mutex_lock@PLT    # 1. Lock mutex
    movl    pings(%rip), %eax         # 2. Load pings
    addl    $1, %eax                  # 3. Increment
    movl    %eax, pings(%rip)         # 4. Store back
    leaq    mutex(%rip), %rax
    movq    %rax, %rdi
    call    pthread_mutex_unlock@PLT  # 5. Unlock mutex
```
The mutex ensures atomicity of the entire increment operation.

## Real-World Race Condition Examples

### Example 1: Lost Update
```
Thread 1: reads 100
Thread 2: reads 100
Thread 1: increments to 101
Thread 2: increments to 101
Thread 1: writes 101
Thread 2: writes 101
```
Result: Only one increment is counted instead of two.

### Example 2: Cache Line Bouncing
```
Thread 1: reads 110
Thread 2: reads 101
Thread 1: increments to 111
Thread 2: increments to 102
Thread 1: writes 111
Thread 2: writes 102
```
Result: One thread's update is lost due to cache inconsistency.

## Performance Impact

1. **Without Mutex**:
   - Faster execution
   - Inconsistent results
   - Values can be less than or more than expected
   - Example: 100,000 iterations might give 150,000 or 80,000

2. **With Mutex**:
   - Slower execution due to lock/unlock overhead
   - Consistent results
   - Always gives expected value
   - Example: 100,000 iterations will always give 200,000

## Debugging Tips

1. **Identify Race Conditions**:
   - Look for shared resources
   - Check for non-atomic operations
   - Monitor for inconsistent results

2. **Common Symptoms**:
   - Inconsistent final values
   - Values less than expected (lost updates)
   - Values more than expected (cache issues)
   - Different results on different runs

3. **Verification**:
   - Run multiple times
   - Check assembly code
   - Use thread sanitizers
   - Monitor cache behavior

## Integer vs Float Assembly Comparison

| Aspect | Integer | Float | Impact |
|--------|---------|-------|---------|
| **Registers** | `%eax`, `%rbp` | `%xmm0`, `%xmm1` | Float uses SSE registers |
| **Constants** | Immediate (`$1`) | Memory (`.LC1`) | Float needs memory load |
| **Increment** | `addl $1, -4(%rbp)` | `movss`, `addss`, `movss` | Float needs 3 instructions |
| **Compare** | `cmpl $99999, -4(%rbp)` | `comiss -4(%rbp), %xmm0` | Float uses SSE compare |
| **Memory Ops** | 1 per operation | 3-4 per operation | Float has more cache pressure |
| **Race Risk** | Medium | High | Float ops take longer |
| **Debugging** | Straightforward | Complex (SSE + IEEE 754) | Float needs more care |
| **Performance** | Faster | Slower | Float has more overhead |

### Key Points
- Float operations are inherently more complex due to SSE requirements
- More memory operations increase race condition risk
- IEEE 754 format adds complexity to comparisons
- Proper synchronization is more critical for float operations
