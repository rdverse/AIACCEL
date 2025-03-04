#include <stdio.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

struct print_message_store{
    const char* message;
    int number;
};


void *thread_function(void *args ){

    struct print_message_store *p = (struct print_message_store *)args;

    for(int i=0;i<p->number;i++){
        printf("%s %d\n", p->message, i);
    }
    return NULL;
}


int main(){
    pthread_t thread1, thread2;
    struct print_message_store pmsm1 = {"Hello", 3};
    struct print_message_store pmsm2 = {"World", 5};

    if (pthread_create(&thread1, NULL, thread_function, &pmsm1)!=0){
        perror("Failed to create thread1");
        exit(1);
    }

    if (pthread_create(&thread2, NULL, thread_function, &pmsm2)!=0){
        perror("Failed to create thread2");
        exit(1);
    }
    
    // Wait for both threads to finish execution before main thread exits
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    return 0;
    
}