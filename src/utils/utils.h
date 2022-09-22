/*
General convenience functions
*/

#pragma once
#include <tuple>
#include <string>

namespace utils {

void print() {
    std::cout << "\n";
}

template <typename T, typename... Args>
void print(T val, Args... args) {
    std::cout << val << " ";
    print(args...);
}


}