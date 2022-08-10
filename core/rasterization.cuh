#ifndef RENDERER_CORE_RASTERIZATION
#define RENDERER_CORE_RASTERIZATION

#include "CUcommons.hpp"
#include "../classes/vertex_triangle.cuh"
#include <math_constants.h>

//TODO: Optimize/reimplement using assumptions like ordering of vertices
__common__ inline double sign (point3 p1, point3 p2, point3 p3)
{
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

__common__ bool PointInTriangle (point3 pt, point3 v1, point3 v2, point3 v3)
{
    double d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}
//TODO end

__global__ void rasterization_shader(size_t triangle_num, vertex_triangle *triangles,
                                     size_t x_frag_num, size_t y_frag_num, color *fragment_buffer,
                                     point3 *vertex_buffer,
                                     double x_min=-1, double x_max=1, double y_min=-1, double y_max=1){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int frag_i = index; frag_i < x_frag_num * y_frag_num; frag_i += stride){
        size_t y_i = frag_i % y_frag_num;
        size_t x_i = (frag_i - y_i) / x_frag_num;
        double y_pos = ((double) y_i/(double) y_frag_num) * (double)(y_max-y_min) + (double)y_min;
        double x_pos = ((double) x_i/(double) x_frag_num) * (double)(x_max-x_min) + (double)x_min;
        double max_z = CUDART_MAX_NORMAL_F;
        fragment_buffer[frag_i] = color(0,0,0); // init frag buffer

        for (unsigned int triangle_i = 0; triangle_i < triangle_num; ++triangle_i){
            if (PointInTriangle(point3(x_pos, y_pos, 0),
                                triangles[triangle_i].x(vertex_buffer),
                                triangles[triangle_i].y(vertex_buffer),
                                triangles[triangle_i].z(vertex_buffer))){
                if (triangles[triangle_i].x(vertex_buffer).z() < max_z){ //TODO: calculate precise z coordinate
                    max_z = triangles[triangle_i].x(vertex_buffer).z();
                    fragment_buffer[frag_i] = color(1,0,triangle_i); //TODO: color
                }
            }
        }

    }

}



#endif
