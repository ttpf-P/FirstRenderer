#include <iostream>
#include "boost/stacktrace.hpp"
#include "CUcommons.hpp"

#include "core/geometry.cuh"
#include "classes/triangle.cuh"


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

    //-------------------------geometry stage-------------------------
    point3 *vertex_buf_old;
    point3 *vertex_buf_new;

    size_t vertex_num = 3;

    checkCudaErrors(cudaMallocManaged(&vertex_buf_old, sizeof(point3)*vertex_num));
    checkCudaErrors(cudaMallocManaged(&vertex_buf_new, sizeof(point3)*vertex_num));


    for (int i = 0; i < 3; ++i){
        vertex_buf_old[i] = point3::random()+point3(0,0,1);
    }

    std::cout << "initial vertex buffer" << std::endl;
    std::cout << vertex_buf_old[0] << std::endl;
    std::cout << vertex_buf_old[1] << std::endl;
    std::cout << vertex_buf_old[2] << std::endl;

    projection_shader<<<1,3>>>(vertex_num, vertex_buf_old, vertex_buf_new,
                               point3(0,0,0), point3(0,0,1), 1);

    cudaDeviceSynchronize();

    std::cout << "final vertex buffer";
    std::cout << vertex_buf_new[0] << std::endl;
    std::cout << vertex_buf_new[1] << std::endl;
    std::cout << vertex_buf_new[2] << std::endl;

    cudaFree(vertex_buf_old);

    //-------------------------rasterization stage-------------------------

    cudaFree(vertex_buf_new);



    cudaDeviceReset();

    return 0;
}

