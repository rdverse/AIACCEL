#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_map>
// naive 
// optimized

bool is_anagram(std::vector<char> arr1, std::vector<char> arr2){

    std::unordered_map<char, int> store;

    for(int item : arr1){
       store[item];  
    }

    for(int item2 : arr2){
        if (store.count(item2)){
          store[item2]--;
        }
        else{
            return false;
        }
    }
    return true;
}

int main(){
    std::vector<char> a = {'a', 'n', 'a', 'g', 'r', 'a', 'm'};
    std::vector<char> b = {'n', 'a', 'g', 'a', 'r', 'a', 'm'};
    
    std::vector<char> c = {'a', 'n', 'a', 'g', 'r', 'a', 'm'};
    std::vector<char> d = {'b', 'z', 'y', 'x', 'w', 'v', 'z'};

    std::cout << "is anagram a and b?" << is_anagram(a,b)<<std::endl; 
    std::cout << "is anagram a and b?" << is_anagram(c,d); 

    return 0;
}