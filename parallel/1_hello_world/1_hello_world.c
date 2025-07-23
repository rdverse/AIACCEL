#include <stdio.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

void *thread_function(void *args ){
    for(int i=0;i<10;i++){
        printf("This is Hello world %d\n", i);
    }
}

int main(){
    pthread_t thread1, thread2;

    if (pthread_create(&thread1, NULL, thread_function, NULL)!=0){
        perror("Failed to create thread1");
        exit(1);
    }
    if (pthread_create(&thread2, NULL, thread_function, NULL)!=0){
        perror("Failed to create thread2");
        exit(1);
    }
    // to ensure both thread1 and thread2 run asynchronously
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    return 0;
}