#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

/*
Most basic intro into running multi-thread application.
*/

struct data{
    int num;
};

void* basic_func(void* arg){
    data* d = (data*)arg;
    printf("test %d\n", d->num);
    sleep(4); // to show that our threads run in parrallel 
    printf("test %d\n", d->num);
    return NULL;
}

int main() {
    pthread_t t1,t2;
    data th1 = {1};
    data th2 = {2};

    if (pthread_create(&t1, NULL, &basic_func, &th1) != 0) {
        perror("Error creating thread1");
        return 1;
    }
    if (pthread_create(&t2, NULL, &basic_func, &th2) != 0) {
        perror("Error creating thread2");
        return 1;
    }
    
    if (pthread_join(t1, NULL) != 0) {
        perror("Error joining thread1");
        return 1;
    }
    if (pthread_join(t2, NULL) != 0) {
        perror("Error joining thread2");
        return 1;
    }

    return 0;
}