#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>

int pings = 0;

pthread_mutex_t mutex;

void* routine(void* arg){
    for (int i=0;i<100000;i++){
    pthread_mutex_lock(&mutex);
    pings++;
    pthread_mutex_unlock(&mutex);
    }
    return NULL;
}


int main(int argc, char* argv[]){
    pthread_t t1,t2;

    if (pthread_mutex_init(&mutex, NULL)!=0){
        perror("Mutex init failes");
    }

    if(pthread_create(&t1, NULL, &routine, NULL)!=0){
        return 1;
    }
if (pthread_create(&t2, NULL, &routine, NULL)!=0){
        return 2;
    }
    
if (pthread_join(t1, NULL)!=0){
        return 3;
    }
if (pthread_join(t2, NULL)!=0){
        return 4;
    }

    printf("Number of pings : %d\n", pings);

   if(pthread_mutex_destroy(&mutex)){
        perror("Mutex destroy failed");
   }

return 0;

}