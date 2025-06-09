# Producer-Consumer Problem Implementation

This is an implementation of the classic Producer-Consumer problem using POSIX threads (pthreads) in C++. The implementation demonstrates proper thread synchronization using mutexes and condition variables.

## Problem Statement

The Producer-Consumer problem is a classic synchronization problem where:

- **Producers**: Add items to a shared buffer
- **Consumers**: Remove items from the shared buffer
- **Buffer**: A fixed-size shared resource that can be full or empty

## Key Components

1. **Shared Buffer**
   - Fixed-size array (10 elements)
   - Counter to track number of items
   - Protected by mutex

2. **Synchronization Mechanisms**
   - Mutex: Protects the shared buffer
   - Condition Variables:
     - `not_full`: Signals when buffer has space
     - `not_empty`: Signals when buffer has items

3. **Thread Management**
   - Multiple producer threads
   - Multiple consumer threads
   - Proper thread creation and joining

## Implementation Details

### Buffer Structure
```cpp
struct buffer {
    int data[10];
    int count = 0;   
};
```

### Synchronization
```cpp
static pthread_mutex_t mutex;        // Protects buffer access
static pthread_cond_t not_full;      // Signals when buffer has space
static pthread_cond_t not_empty;     // Signals when buffer has items
```

### Producer Thread
```cpp
while (true) {
    pthread_mutex_lock(&mutex);
    while (buf.count >= 10) {
        pthread_cond_wait(&not_full, &mutex);
    }
    // Add to buffer
    pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&not_empty);
    sleep(1);  // Outside mutex lock
}
```

### Consumer Thread
```cpp
while (true) {
    pthread_mutex_lock(&mutex);
    while (buf.count == 0) {
        pthread_cond_wait(&not_empty, &mutex);
    }
    // Remove from buffer
    pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&not_full);
    sleep(1);  // Outside mutex lock
}
```

## Key Learnings

1. **Condition Variables vs Sleep**
   - `pthread_cond_wait`: 
     - Efficient: Thread sleeps until condition is met
     - Synchronized: Works with mutex for proper synchronization
     - Immediate: Wakes up as soon as condition changes
   - `sleep`:
     - Inefficient: Sleeps for fixed duration
     - Unsynchronized: Can miss condition changes
     - Fixed delay: Must wait full duration

2. **Mutex Lock Duration**
   - Keep critical sections as short as possible
   - Never sleep while holding a mutex
   - Release mutex before any blocking operations

3. **Condition Variable Usage**
   - Always use while loop for condition checks
   - Signal after releasing mutex
   - Wait inside mutex lock

4. **Thread Safety**
   - All shared resources must be protected
   - Use proper synchronization primitives
   - Consider deadlock prevention

## Common Pitfalls

1. **Sleep Inside Mutex**
   - Blocks other threads unnecessarily
   - Reduces parallelism
   - Can cause deadlocks

2. **Missing Condition Checks**
   - Using if instead of while
   - Can miss condition changes
   - Leads to race conditions

3. **Improper Signal Timing**
   - Signaling before unlocking mutex
   - Signaling wrong condition
   - Missing signals

## Best Practices

1. **Synchronization**
   - Use condition variables for thread communication
   - Keep mutex locks as short as possible
   - Always check conditions in while loops

2. **Error Handling**
   - Check return values of pthread functions
   - Proper cleanup of resources
   - Handle thread creation failures

3. **Performance**
   - Minimize critical sections
   - Avoid busy waiting
   - Use appropriate sleep durations

## Performance Monitoring

The implementation includes a comprehensive monitoring system that tracks various performance metrics:

### Monitor Class
- Singleton pattern for global access
- Thread-safe statistics collection
- Real-time performance metrics

### Metrics Tracked
1. **Overall Statistics**
   - Throughput (operations per second)
   - Average latency (microseconds)
   - Buffer utilization (percentage)
   - Active thread count

2. **Buffer Statistics**
   - Maximum items in buffer
   - Number of times buffer was full
   - Number of times buffer was empty
   - Total operations performed

3. **Thread Statistics**
   - Per-thread operation count
   - Per-thread average latency
   - Thread type (producer/consumer)
   - Last operation timestamp

### Usage
```cpp
// Get monitor instance
Monitor* monitor = Monitor::getInstance();

// Start monitoring
monitor->start();

// Record an operation
monitor->recordOperation(thread_id, "producer", latency, buffer_count);

// Print statistics
monitor->printStats();

// Stop monitoring
monitor->stop();
```

### Building with Monitoring
```bash
g++ -pthread producer_consumer.cc monitor.cc -o producer_consumer
```

## Building and Running

```bash
g++ -pthread producer_consumer.cc -o producer_consumer
./producer_consumer
```

## Output Format

The program outputs:
- Producer thread ID and value added
- Consumer thread ID and value removed
- Current buffer count

Example:
```
Producer 1234567890: put 42 (count: 1)
Consumer 0987654321: got 42 (count: 0)
``` 