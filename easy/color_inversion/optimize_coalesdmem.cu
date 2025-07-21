#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void invert_kernel(uchar4* image, int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;

    uchar4 pixel = image[idx];
    pixel.x = 255 - pixel.x;
    pixel.y = 255 - pixel.y;
    pixel.z = 255 - pixel.z;
    image[idx] = pixel;
}

void solve(unsigned char* image, int width, int height) {
    int pixel_count = width * height;

    uchar4* d_image;
    hipMalloc(&d_image, pixel_count * sizeof(uchar4));
    hipMemcpy(d_image, image, pixel_count * sizeof(uchar4), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (pixel_count + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, pixel_count);
    hipDeviceSynchronize();

    hipMemcpy(image, d_image, pixel_count * sizeof(uchar4), hipMemcpyDeviceToHost);
    hipFree(d_image);
}


int main(){
    int width, height;
    std::cin >> width >> height;

    std::vector<unsigned char> image(width * height * 4);
    
    for (int i = 0; i < width * height * 4; ++i){
        int temp;
        std::cin >> temp;
        image[i] = (unsigned char)temp;  
    }

    solve(image.data(), width, height);

    for(int i = 0; i < width * height; ++i){
        for(int j = 0; j < 4; ++j){
            std::cout << (int)image[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}