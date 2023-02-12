/*
Entrance file for the entire operation, contains only the logic for
the standalone executable portion. Should not include libtorch for
faster compile times.
*/

#include <iostream>
//#include "utils/utils.h"
//#include "net-v1/model.h"
#include <torch/cuda.h>
#include <torch/torch.h>

//namespace ut = utils;


int main() {
    int a = torch::cuda::device_count();
    std::cout << torch::cuda::is_available() << torch::cuda::cudnn_is_available();
}