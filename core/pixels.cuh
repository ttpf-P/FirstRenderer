#ifndef RENDERER_CORE_PIXELS
#define RENDERER_CORE_PIXELS

__global__ void fragment_to_pixel_shader(size_t frag_num, color *frag_buf, uchar3 *pixel_buf){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < frag_num; i += stride){
        pixel_buf[i].x = (unsigned char)(frag_buf[i].x()*255);
        pixel_buf[i].y = (unsigned char)(frag_buf[i].y()*255);
        pixel_buf[i].z = (unsigned char)(frag_buf[i].z()*255);
        /*pixel_buf[i].x = 255;
        pixel_buf[i].y = 0;
        pixel_buf[i].z = i % 255;*/
    }
}

#endif
