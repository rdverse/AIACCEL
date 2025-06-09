#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <random>
#define THREAD_COUNT 2

/*
Problem Statement:

Producer:
- adds to buffer
- buffer is full

Consumer:
- fetches data to process from buffer
- buffer is empty

A shared buffer in memory for multiple threads:
- Producer adds to buffer
- Consumer fetches data to process from buffer
- mutex for shared memory
- check if buffer is full
- multi-producer, single consumer (buffer empty)
*/

struct buffer {
    int data[10];
    int count=0;   
};

class ThreadPT {
    static buffer buf;  // Shared buffer
    static pthread_mutex_t mutex;  // Mutex to protect the buffer
    static pthread_cond_t not_full;  // Condition for buffer not full
    static pthread_cond_t not_empty; // Condition for buffer not empty
    
public:
    static void* producer(void* args) {
        while (true) {
            int a = rand() % 100;
            
            pthread_mutex_lock(&mutex);
            // Wait if buffer is full
            while (buf.count >= 10) {
                pthread_cond_wait(&not_full, &mutex);
            }
            
            // Add to buffer
            buf.data[buf.count] = a;
            buf.count++;
            printf("Producer %lu: put %d (count: %d)\n", pthread_self(), a, buf.count);
            
            pthread_mutex_unlock(&mutex);
            pthread_cond_signal(&not_empty);  // Signal that buffer is not empty
            sleep(1);  // Sleep outside mutex lock
        }
        return NULL;
    }

    static void* consumer(void* args) {
        while (true) {
            pthread_mutex_lock(&mutex);
            // Wait if buffer is empty
            while (buf.count == 0) {
                pthread_cond_wait(&not_empty, &mutex);
            }
            
            int b = buf.data[buf.count-1];
            buf.count--;
            printf("Consumer %lu: got %d (count: %d)\n", pthread_self(), b, buf.count);
            
            pthread_mutex_unlock(&mutex);
            pthread_cond_signal(&not_full);  // Signal that buffer is not full
            sleep(1);  // Sleep outside mutex lock
        }
        return NULL;
    }

    static void init() {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&not_full, NULL);
        pthread_cond_init(&not_empty, NULL);
    }

    static void cleanup() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&not_full);
        pthread_cond_destroy(&not_empty);
    }
};

// Define static members
buffer ThreadPT::buf;
pthread_mutex_t ThreadPT::mutex;
pthread_cond_t ThreadPT::not_full;
pthread_cond_t ThreadPT::not_empty;

int main(int argc, char* argv[]) {
    srand(time(NULL));
    pthread_t th[THREAD_COUNT];
    
    ThreadPT::init();  // Initialize mutex and condition variables

    for (int i=0; i<THREAD_COUNT; i++) {
        if (i%2==0) {
            if (pthread_create(&th[i], NULL, &ThreadPT::producer, NULL) != 0) {
                perror("Unable to launch producer");
            }
        } else {
            if(pthread_create(&th[i], NULL, &ThreadPT::consumer, NULL) != 0) {
                perror("Unable to launch consumer");
            }
        }
    }
    for (int i=0; i<THREAD_COUNT; i++) {
        if (pthread_join(th[i], NULL) != 0) {
            perror("Unable to join threads");
        }
    }

    ThreadPT::cleanup();  // Cleanup mutex and condition variables

    return 0;
}