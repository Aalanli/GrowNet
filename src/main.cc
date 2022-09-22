#include <iostream>

#include "utils/torch_utils.h"
#include "utils/utils.h"
#include "lib.h"


int main() {
    auto a = utils::test_fn(4);
    utils::print(a);
}