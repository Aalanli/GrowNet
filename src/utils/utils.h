/*
General convenience functions
*/

#pragma once
#include <tuple>
#include <string>
#include <vector>

namespace utils {

void print() {
    std::cout << "\n";
}

template <typename T, typename... Args>
void print(T val, Args... args) {
    std::cout << val << " ";
    print(args...);
}


int prod(const std::vector<int> &a) {
    int i = 1;
    for (auto const &it : a) {
        i *= it;
    }
    return i;
}


}