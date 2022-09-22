#include <torch/torch.h>
#include <iostream>

#include <utils.h>

int main() {
    auto a = torch::rand({4, 4});
    utils::print(a);
}