#ifndef RENDERER_CLASSES_TRIANGLE
#define RENDERER_CLASSES_TRIANGLE

#include "CUcommons.hpp"

class vertex_triangle {
public:
    __common__ vertex_triangle() : vertices{0,0,0} {};
    __common__ vertex_triangle(size_t v0, size_t v1, size_t v2) : vertices{v0, v1, v2} {};
    __common__ point3 x(point3 *vertex_buffer){return vertex_buffer[vertices[0]];}
    __common__ point3 y(point3 *vertex_buffer){return vertex_buffer[vertices[1]];}
    __common__ point3 z(point3 *vertex_buffer){return vertex_buffer[vertices[2]];}
public:
    size_t vertices[3];
};


#endif
