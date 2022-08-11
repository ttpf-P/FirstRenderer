#ifndef RENDERER_CORE_GEOMETRY
#define RENDERER_CORE_GEOMETRY

#include "CUcommons.hpp"

__global__ void projection_shader(const size_t vertex_num, point3 *vertices_3d, point3 *vertices_projected, const double screen_dist_to_origin = 1){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    const point3 screen_unit_normal = point3(0,0,1);

    for (unsigned int i = index; i < vertex_num; i += stride){
        vec3 unit_direction = unit_vector(-vertices_3d[i]);
        double s = sum(screen_unit_normal * unit_direction);
        if (s != 0){ //TODO: check if branching at the end and ignoring zero division is faster (probably not)
            double dist = (screen_dist_to_origin - sum(screen_unit_normal*vertices_3d[i])) / s;
            vec3 intersect = vertices_3d[i] + unit_direction*dist;
            vertices_projected[i] = vec3(intersect.x(), intersect.y(), dist); //TODO: implement camera rotation (or rotate everything around the camera)

        }
    }

}

#endif
