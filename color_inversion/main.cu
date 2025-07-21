#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx >= width * height) return;

    int baseidx = idx * 4;

    image[baseidx] = 255 - image[baseidx];
    image[baseidx + 1] = 255 - image[baseidx + 1];
    image[baseidx + 2] = 255 - image[baseidx + 2];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {

    unsigned char *d_image;

    // Allocate GPU memory
    hipMalloc(&d_image, width * height * 4 * sizeof(unsigned char));

    // Copy data from host to device
    hipMemcpy(d_image, image, width * height * 4 * sizeof(unsigned char), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    hipDeviceSynchronize();

    hipMemcpy(image, d_image, width * height * 4 * sizeof(unsigned char), hipMemcpyDeviceToHost);

    hipFree(d_image);
}

int main(){
    int width, height;
    std::cin >> width >> height;

    // Read input vectors from standard input
    std::vector<unsigned char> image(width * height * 4);
    
    for (int i = 0; i < width * height * 4; ++i){
        int temp;
        std::cin >> temp;
        image[i] = (unsigned char)temp;  // Convert from int to unsigned char
    }

    // Call the solve function
    solve(image.data(), width, height);

    // Output the resulting vector
    for(int i = 0; i < width * height; ++i){
        for(int j = 0; j < 4; ++j){
            std::cout << (int)image[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}