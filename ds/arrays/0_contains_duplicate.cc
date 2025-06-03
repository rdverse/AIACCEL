#include <stdio.h>
#include <iostream>
#include <unordered_set>


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
    }
    return false;
}


int main(){

int test2[] = {1, 2, 5, 5};
int test1[] = {1, 2, 3, 4};
int size = sizeof(test1) / sizeof(test1[0]);

std::cout<< "Naive output test1: "<< naive(test1, size) << std::endl;
std::cout<< "Optimized output test1: "<< naive(test1, size) <<std::endl;

std::cout<< "Optimized output test2 : "<< naive(test2, size)<< std::endl;
std::cout<< "Optimized output test2: "<< naive(test2, size)<< std::endl;

return 0;    
}