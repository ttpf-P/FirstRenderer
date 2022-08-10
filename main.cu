#include <iostream>
#include "boost/stacktrace.hpp"
#include "CUcommons.hpp"

#include "core/geometry.cuh"
#include "core/rasterization.cuh"
#include "classes/vertex_triangle.cuh"

const unsigned long long state2 = 88172645463325252ULL;

int main() {
    std::cerr << "Hello, World!" << std::endl;
    std::cerr << boost::stacktrace::stacktrace() << std::endl;

    //-------------------------init stage-------------------------
    // constants
    size_t vertex_num = 4;
    size_t triangle_num = 2;

    size_t x_frag_num = 128;
    size_t y_frag_num = 128;

    // buffer declaration
    point3 *vertex_buf_old;
    point3 *vertex_buf_new;

    vertex_triangle *triangle_buf_old;
    vertex_triangle *triangle_buf_new;

    color *fragment_buf;

    // buffer allocation
    checkCudaErrors(cudaMallocManaged(&vertex_buf_old, sizeof(point3)*vertex_num));
    checkCudaErrors(cudaMallocManaged(&vertex_buf_new, sizeof(point3)*vertex_num));

    checkCudaErrors(cudaMallocManaged(&triangle_buf_old, sizeof(vertex_triangle)*triangle_num));
    checkCudaErrors(cudaMallocManaged(&triangle_buf_new, sizeof(vertex_triangle)*triangle_num));

    checkCudaErrors(cudaMallocManaged(&fragment_buf, sizeof(color)*x_frag_num*y_frag_num));

    // fill buffers for testing
    for (int i = 0; i < vertex_num; ++i){
        vertex_buf_old[i] = point3::random(-1,1)+point3(0,0,2);
    }

    triangle_buf_old[0] = vertex_triangle(0,1,2);
    triangle_buf_old[1] = vertex_triangle(1,2,3);

    //-------------------------geometry stage-------------------------
    std::cerr << "initial vertex buffer" << std::endl;
    std::cerr << vertex_buf_old[0] << std::endl;
    std::cerr << vertex_buf_old[1] << std::endl;
    std::cerr << vertex_buf_old[2] << std::endl;
    std::cerr << vertex_buf_old[3] << std::endl;

    projection_shader<<<1,3>>>(vertex_num, vertex_buf_old, vertex_buf_new,
                               point3(0,0,0), point3(0,0,1), 1);
    cudaDeviceSynchronize();

    std::cerr << "final vertex buffer" << std::endl;
    std::cerr << vertex_buf_new[0] << std::endl;
    std::cerr << vertex_buf_new[1] << std::endl;
    std::cerr << vertex_buf_new[2] << std::endl;
    std::cerr << vertex_buf_new[3] << std::endl;

    //-------------------------rasterization stage-------------------------
    rasterization_shader<<<128*4,32>>>(triangle_num, triangle_buf_old,
                                       x_frag_num, y_frag_num, fragment_buf,
                                       vertex_buf_new);
    cudaDeviceSynchronize();


    //-------------------------output stage-------------------------
    std::cout << "P3\n" << x_frag_num << " " << y_frag_num << "\n255\n";
    for (long long y = y_frag_num-1; y >= 0 ; --y) {
        for (size_t x = 0; x < x_frag_num; x++){
            std::cout << (int) (fragment_buf[x_frag_num*y+x].x()*255) << " "
                 << (int) (fragment_buf[x_frag_num*y+x].y()*255) << " "
                 << (int) (fragment_buf[x_frag_num*y+x].z()*255) << "\n";
        }
    }

    //-------------------------cleanup stage-------------------------
    cudaFree(vertex_buf_old);
    cudaFree(vertex_buf_new);

    cudaFree(triangle_buf_old);
    cudaFree(triangle_buf_new);

    cudaFree(fragment_buf);

    cudaDeviceReset();

    return 0;
}

