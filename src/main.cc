/*
Entrance file for the entire operation, contains only the logic for
the standalone executable portion. Should not include libtorch for
faster compile times.
*/

#include <iostream>
#include "utils/utils.h"
#include "net-v1/model.h"

namespace ut = utils;

int main() {
    Tensor<float> test({1, 2, 3});
    ut::print(test.sizes[0]);
}