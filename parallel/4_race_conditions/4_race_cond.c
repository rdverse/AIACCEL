#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>

int pings = 0;


void* routine(void* arg){
    for (int i=0;i<100000;i++){
    pings++;
    }

    /*
    case1:
    ops        t1     t2
    read         100    101  
    increment    100    101
    write        101    102

    case2 issue:
    ops        t1     t2
    read       110    101  
    increment  110    111
    write      111    102 ****
    */ 

    return NULL;
}



int main(int argc, char* argv[]){
    pthread_t t1,t2;

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


return 0;

}