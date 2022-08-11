#include <iostream>

#include "glew.h"
#include "glut.h"
#include "cuda_gl_interop.h"

#include "CUcommons.hpp"

#include "core/geometry.cuh"
#include "core/rasterization.cuh"
#include "core/pixels.cuh"
#include "classes/vertex_triangle.cuh"

const unsigned long long state2 = 88172645463325252ULL;
const size_t x_frag_num = 1920;
const size_t y_frag_num = 1080;

const size_t vertex_num = 4;
const size_t triangle_num = 2;

// buffer declaration
point3 *vertex_buf_old;
point3 *vertex_buf_new;

vertex_triangle *triangle_buf_old;
vertex_triangle *triangle_buf_new;

color *fragment_buf;

GLuint pbo = 0; //pixel buffer
GLuint tex = 0; //texture buffer
cudaGraphicsResource *cuda_pbo_resource;

//resizing vars
double x_min = -1;
double x_max = 1;
double y_min = x_min*((double)y_frag_num/x_frag_num);
double y_max = x_max*((double)y_frag_num/x_frag_num);



void render_frame(uchar3 *pixel_buf){
    //-------------------------init stage-------------------------


    //-------------------------geometry stage-------------------------

    projection_shader<<<1,4>>>(vertex_num, vertex_buf_old, vertex_buf_new);
    checkCudaErrors(cudaDeviceSynchronize());

    //-------------------------rasterization stage-------------------------
    rasterization_shader<<<2048*32,32>>>(triangle_num, triangle_buf_old,
                                       x_frag_num, y_frag_num, fragment_buf,
                                       vertex_buf_new,
                                       x_min, x_max, y_min, y_max);
    checkCudaErrors(cudaDeviceSynchronize());


    //-------------------------output stage-------------------------
    fragment_to_pixel_shader<<<2048*32,32>>>(x_frag_num*y_frag_num, fragment_buf, pixel_buf);
    checkCudaErrors(cudaDeviceSynchronize());
    /*std::cout << "P3\n" << x_frag_num << " " << y_frag_num << "\n255\n";
    for (long long y = y_frag_num-1; y >= 0 ; --y) {
        for (size_t x = 0; x < x_frag_num; x++){
            std::cout << (int) (fragment_buf[x_frag_num*y+x].x()*255) << " "
                      << (int) (fragment_buf[x_frag_num*y+x].y()*255) << " "
                      << (int) (fragment_buf[x_frag_num*y+x].z()*255) << "\n";
        }
    }*/

}

void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, x_frag_num*y_frag_num*sizeof(uchar3), nullptr,
                 GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                      cudaGraphicsMapFlagsWriteDiscard);
    }

void processNormalKeys(unsigned char key, int x, int y) {

    if (key == 27)
        exit(0);
}

void gl_wrapper(){

    uchar3 *pixel_buf;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&pixel_buf, nullptr,
                                         cuda_pbo_resource));
    render_frame(pixel_buf);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource));

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, x_frag_num, y_frag_num, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_POLYGON);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1, -1);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1, 1);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1, 1);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1, -1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}

int main(int argc, char **argv) {

    //render_frame();

    //-------------------------window init-------------------------
    glutInit(&argc, argv);
    glutInitWindowPosition(-1,-1);
    glutInitWindowSize(x_frag_num,y_frag_num);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    gluOrtho2D(0, x_frag_num, y_frag_num, 0);
    glutCreateWindow("TestOutput");
    glutFullScreen();
    glewInit();

    glutDisplayFunc(gl_wrapper);
    glutKeyboardFunc(processNormalKeys);

    //-------------------------buffer allocation/initialization-------------------------
    checkCudaErrors(cudaMallocManaged(&vertex_buf_old, sizeof(point3)*vertex_num));
    checkCudaErrors(cudaMallocManaged(&vertex_buf_new, sizeof(point3)*vertex_num));

    checkCudaErrors(cudaMallocManaged(&triangle_buf_old, sizeof(vertex_triangle)*triangle_num));
    checkCudaErrors(cudaMallocManaged(&triangle_buf_new, sizeof(vertex_triangle)*triangle_num));

    checkCudaErrors(cudaMallocManaged(&fragment_buf, sizeof(color)*x_frag_num*y_frag_num));

    initPixelBuffer();

    // fill buffers for testing
    for (int i = 0; i < vertex_num; ++i){
        vertex_buf_old[i] = point3::random(-1,1)+point3(0,0,2);
    }

    triangle_buf_old[0] = vertex_triangle(0,1,2);
    triangle_buf_old[1] = vertex_triangle(1,2,3);

    //-------------------------main loop-------------------------
    glutMainLoop();
    //-------------------------cleanup-------------------------
    checkCudaErrors(cudaFree(vertex_buf_old));
    checkCudaErrors(cudaFree(vertex_buf_new));

    checkCudaErrors(cudaFree(triangle_buf_old));
    checkCudaErrors(cudaFree(triangle_buf_new));

    checkCudaErrors(cudaFree(fragment_buf));

    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
        }
    return 0;
}

