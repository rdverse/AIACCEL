#include<stdio.h>
#include<stdlib.h>

int add(int a, int b){
    return a + b;
}

// int main(){
//     return 10;
// }

int main(int argc, char **argv){
    int z = 10;
    int x = add(10, argc); 
    printf("argc + 10: %d\n", x);

    int sum = 0; 
    
    // for (int i = 0; i < argc; i++){
    //     sum = sum +  atoi(argv[i]);
    // }
    sum = sum + atoi(argv[1]);
    sum = atoi(argv[2]) + atoi(argv[1]);
    sum = add(8 , 10);
    printf("sum: %d\n", sum);

    return 0;
}
