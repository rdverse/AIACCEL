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

*/

/*
A shared buffer in memory for multiple threads

Producer adds to buffer,
Consumer fetches data to process from buffer

mutex for shared memory
check if buffer is full
multi-producer, single consumer (buffer empty)

*/


struct buffer {
    int data[10];
    int count=0;   
};

class ThreadPT {
    static buffer buf;  // Shared buffer
    static pthread_mutex_t mutex;  // Mutex to protect the buffer
    
public:
    static void* producer(void* args) {
        while (true){

        int a = rand() % 100;
        pthread_mutex_lock(&mutex);
        // Add to buffer
        buf.data[buf.count] = a;
        buf.count++;
        printf("put %d\n", a);
        pthread_mutex_unlock(&mutex);
        }
        return NULL;
    }

    static void* consumer(void* args) {

        while (true){
        pthread_mutex_lock(&mutex);
        int b = buf.data[buf.count-1];
        buf.count--;
        printf("Catch %d\n", b);
        sleep(1);
        pthread_mutex_unlock(&mutex);
        }
        return NULL;
    }

    static void init() {
        pthread_mutex_init(&mutex, NULL);
    }

    static void cleanup() {
        pthread_mutex_destroy(&mutex);
    }
};

// Define static members
buffer ThreadPT::buf;
pthread_mutex_t ThreadPT::mutex;

int main(int argc, char* argv[]) {
    srand(time(NULL));
    pthread_t th[THREAD_COUNT];
    
    ThreadPT::init();  // Initialize mutex

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

    ThreadPT::cleanup();  // Cleanup mutex

    return 0;
}