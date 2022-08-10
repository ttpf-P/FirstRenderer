#ifndef RENDERER_CORE_GEOMETRY
#define RENDERER_CORE_GEOMETRY

#include "CUcommons.hpp"

__global__ void projection_shader(size_t vertex_num, point3 *vertices_3d, point3 *vertices_projected,
                                  point3 camera_center, vec3 screen_unit_normal, double screen_dist_origin){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < vertex_num; i += stride){
        vec3 unit_direction = unit_vector(camera_center - vertices_3d[i]);
        double s = sum(screen_unit_normal * unit_direction);
        if (s != 0){ //TODO: check if branching at the end and ignoring zero division is faster
            double dist = (screen_dist_origin - sum(screen_unit_normal*vertices_3d[i])) / s;
            if (dist > 0){
                vec3 intersect = vertices_3d[i] + unit_direction*dist;
                vertices_projected[i] = vec3(intersect.x(), intersect.y(), dist); //TODO: implement camera rotation
            }
        }
    }

}

#endif
