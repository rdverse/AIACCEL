#include<stdlib.h>
#include<stdio.h>
#include<pthread.h>

int total_count = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

struct accumulate_args{
    int start;
    int end;
};

void *accumulate(void *args){

    struct accumulate_args *p = (struct accumulate_args*)args;
    
    printf("Start: %d\n", p->start);
    printf("End: %d\n", p->end);
    for(int i=p->start;i<p->end;i++){
    //printf("i and tc: %d %d\n", i, total_count);
    // pthread_mutex_lock(&lock);
    // total_count+=1;
    // pthread_mutex_unlock(&lock); // if lock and unlock are not present race condition occurs

    // For single line operations we can use atomic operations
     __sync_fetch_and_add(&total_count, 1);
    }
}


int main(){
    pthread_t thread1, thread2;
    struct accumulate_args aa1 = {0, 100000000}; // small loops may not produce data race as the scheduler may time slice
    struct accumulate_args aa2 = {0, 100000000}; // using large number so we can see the race condition if mutex (mutual execution) is not used
    // race condition : when two threads try to access the same memory location at the same time
    // this can be avoided by using mutexes

    if (pthread_create(&thread1, NULL, accumulate, &aa1)!=0){
        perror("Failed to create thread1");
        exit(1);
    }

    if (pthread_create(&thread2, NULL, accumulate, &aa2)!=0){
        perror("Failed to create thread2");
        exit(1);
    }
    
    // Wait for both threads to finish execution before main thread exits
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    printf("Total count: %d\n", total_count);
    return 0;
}
