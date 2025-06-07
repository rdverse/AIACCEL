// switching gears and moving to c++
#include<cstdio>
#include<thread>
#include<string>
#include<iostream>

// no need of struct
// struct print_message_store{
//     const char* message;
//     int number;
// };


void thread_function_to_print(int number, std::string message){
    for(int i=0;i<number;i++){
                std::cout << message << " " << i << std::endl;

    }
}

int main(){
    std::string message1 = "Hello";
    std::string message2 = "World";

    std::thread thread1(thread_function_to_print, 10000, message1);
    std::thread thread2(thread_function_to_print, 10000, message2);

    // Wait for both threads to finish execution before main thread exits
    thread1.join();
    thread2.join();

    return 0; 
}