/*
pytorch convenience functions, for separation between functions which includes torch, since
building with it is extremely slow
*/

#pragma once
#include <string>
#include <unordered_map>
#include <torch/types.h>

namespace utils {
    auto like_tensor(const torch::Tensor &a) {
        auto opt = torch::TensorOptions().device(a.device()).dtype(a.dtype());
        return opt;
    }

    inline int n_elements(const torch::Tensor &a) {
        return a.size(0) * a.stride(0);
    }


}

namespace type_repr {
std::string get_runtime_type_str(torch::Dtype t) {
    switch (t) {
        case torch::kInt:
            return std::string("i");
        case torch::kF16:
            return std::string("h");
        case torch::kF32:
            return std::string("f");
        case torch::kF64:
            return std::string("d");
    }
    return std::string("unknown");
}

std::string get_runtime_type_str(int t) {
    return std::to_string(t);
}

std::string construct_runtime_str() {
    return std::string("");
}

template <typename T, typename... Args>
std::string construct_runtime_str(T val, Args... args) {
    return get_runtime_type_str(val) + construct_runtime_str(args...);
}

template <typename... Args>
size_t construct_runtime_id(int ver, Args... args) {
    std::hash<std::string> hasher;
    return hasher(std::to_string(ver) + construct_runtime_str(args...));
}


}