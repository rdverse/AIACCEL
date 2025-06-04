#include <stdio.h>
#include <iostream>
#include <unordered_set>
#include <chrono>

bool naive(int* arr, int size){

    for(int i=0;i<size;i++){
     for(int j=0;j<size;j++){
        if(i==j){
            continue;
        } 
        else{
            if(arr[j]==arr[i]){
                return true;
            }
        }
    }
}   
        

    return false;

} // O(n^2) time O(n) space


bool optimized(int* arr, int size){
    // sort O(nlogn)
    //hashmap O(n) + O(n)
    std::unordered_set<int> store; 
        for(int i=0;i<size;i++){
        if (store.count(arr[i])){
            return true;
        }
        store.insert(arr[i]);
    }
    return false;
} // O(n) time O(n) space


int main(){

int test2[] = {1, 2, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
int test1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
int smoltest1[] = {1, 2, 2, 4};
int smoltest2[] = {1, 2, 3, 4};

int size = sizeof(test1) / sizeof(test1[0]);

//time naive
auto start = std::chrono::high_resolution_clock::now();
for(int i=0;i<100000;i++){
naive(test1, size);
naive(test2, size);

}
std::cout<< "Naive output test1: "<< naive(test1, size) << std::endl;
std::cout<< "Naive output test2: "<< naive(test2, size) <<std::endl;

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout<< "Naive time: "<< duration.count() << std::endl;


auto start4 = std::chrono::high_resolution_clock::now();
for(int i=0;i<100000;i++){
naive(smoltest1, 4);
naive(smoltest2, 4);
}
auto end4 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration4 = end4 - start4;
std::cout<< "Naive time for small arrays: "<< duration4.count() << std::endl;



//time optimized
auto start2 = std::chrono::high_resolution_clock::now();
for(int i=0;i<100000;i++){
optimized(test1, size);
optimized(test2, size);
}
auto end2 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration2 = end2 - start2;
std::cout<< "Optimized time for large arrays: "<< duration2.count() << std::endl;

//time optimized for small arrays
auto start3 = std::chrono::high_resolution_clock::now();
for(int i=0;i<100000;i++){
optimized(smoltest1, 4);
optimized(smoltest2, 4);
}
auto end3 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration3 = end3 - start3;
std::cout<< "Optimized time for small arrays: "<< duration3.count() << std::endl;


std::cout<< "Optimized output test1 : "<< optimized(test1, size)<< std::endl;
std::cout<< "Optimized output test2: "<< optimized(test2, size)<< std::endl;
return 0;    
}


// Interestingly the optimized is slower than the naive for small arrays 
// unordered set is using heap memory and not stack memory
// so it is slower than the naive


// Naive time: 2.29108
// Naive time for small arrays: 0.00418196
// Optimized time for large arrays: 2.1206
// Optimized time for small arrays: 0.144385