#include <iostream>
#include <thread>
#include <chrono>
#include <string>
// g++ threads.cpp -o threads -lpthread
// ./threads
void printy(std::string name) {
    for (int i = 0; i < 5; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "Hello from " << name << std::endl;
    }
}

int main() {
    size_t TEAM_SIZE = 10;
    std::thread team[TEAM_SIZE];
    for (size_t i = 0; i < TEAM_SIZE; i++) {
        team[i] = std::thread(printy, "thread" + std::to_string(i));
    }
    
    for (size_t i = 0; i < TEAM_SIZE; i++) {
        team[i].join();
    }

    return 0;
}