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
    std::thread t1(printy, "thread1");
    std::thread t2(printy, "thread2");

    t1.join();
    t2.join();

    return 0;
}