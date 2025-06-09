#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    int pid = fork(); // creates a child process of current process
    int x = 99;

    if (pid == -1){
        return 1;
    }
    if (pid == 0){
        x++;
    }
    printf("process id %d, xval : %d \n", getpid(), x); // one print from parent and one from childl

    if (pid!=0){
        wait(NULL); // wait for child process to finish
    }

    return 0;
}