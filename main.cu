#include <iostream>
#include "boost/stacktrace.hpp"
#include "CUcommons.hpp"

#include "core/geometry.h"


__global__ void main_kernel(double *state){
    unsigned long long state2 = 88172645463325252ULL;
    auto vec = vec3::random(state2);
    auto vec_2 = vec3::random(state2);
    auto vec_3 = dot(vec,vec_2);
    *state = vec_3;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << boost::stacktrace::stacktrace() << std::endl;
    double *test_state;

    checkCudaErrors(cudaMallocManaged(&test_state, sizeof(double)));
    *test_state = 2;

    main_kernel<<<1,1>>>(test_state);

    cudaDeviceSynchronize();

    std::cout << *test_state << std::endl;

    cudaFree(test_state);
    cudaDeviceReset();

    return 0;
}

