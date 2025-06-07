#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int x = 99;
// threads share memory so issues when modifying and reading at same time
void* routine1(void* args){
    //sleep(15); if this sleep is added then the threads will print different values
    x++;
    sleep(2);
    int id = getpid();
    printf("Thread: process id %d, xval :%d\n", id, x);
    return NULL;
}

void* routine2(void* args){
    int id = getpid();
    sleep(2);
    printf("Thread: process id %d, xval :%d\n", id, x);
    return NULL;
}

int main() {
    pthread_t th1,th2;

    if (pthread_create(&th1, NULL, &routine1, NULL)!=0){
        return 1;
    }
    
    if (pthread_create(&th2, NULL, &routine2, NULL)!=0){
        return 1;
    }

    if (pthread_join(th1, NULL)!=0){
        return 1;
    }
    
    if (pthread_join(th2, NULL)!=0){
        return 1;
    }

    return 0;
}