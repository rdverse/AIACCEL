// So in 1_2, printing hello and world in two threads was causing 38-40 prints from each thread before switch
// This script is to explore why and make it round robin 
#include<cstdio>
#include<thread>
#include<string>
#include<iostream>
#include<mutex>
#include<condition_variable>

std::mutex cout_mutex;
std::condition_variable cv;
bool turn = false;

void thread_function_to_print(int number, std::string message, bool myTurn){
    for(int i=0;i<number;i++){
        // uncomment below line if you want all prints of one thread to be printed once
        // std::lock_guard<std::mutex> lock(cout_mutex);  // Lock

        std::unique_lock<std::mutex> lock(cout_mutex);
        cv.wait(lock, [=] {
            //std::cout<< "Turn: " << turn << " myTurn: " << myTurn << std::endl;
            return turn == myTurn;
        });
        std::cout << message << " " << i << std::endl;
        turn = !turn;
        lock.unlock();
        cv.notify_one();

    }
}

int main(){
    std::string message1 = "Hello";
    std::string message2 = "World";

    // switch to mutexes to see if it changes the behavior
    std::thread thread1(thread_function_to_print, 10000, message1, false);
    std::thread thread2(thread_function_to_print, 10000, message2, true);

    // interesting observations:
    // point at which thread1 and thread2 start interweaving is variable
    // also the last print gets cut off and split - the new thread after
    // context switch probably has higher priority
    // refer to this example - https://en.cppreference.com/w/cpp/thread/condition_variable/wait
    // this code runs at 2.68 and the one without the alternate locks takes 0.87 for 10000 prints!!!!
    // Wait for both threads to finish execution before main thread exits
    thread1.join();
    thread2.join();
    
    return 0; 
}