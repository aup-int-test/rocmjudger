#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= input_size - kernel_size + 1) return;

    output[idx] = 0;
    for(int j = 0; j < kernel_size; ++j){
        output[idx] += input[idx + j] * kernel[j];
    }
    
}

void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {

    int output_size = input_size - kernel_size + 1;

    float *d_input, *d_kernel, *d_output;

    hipMalloc(&d_input, input_size * sizeof(float));
    hipMalloc(&d_kernel, kernel_size * sizeof(float));
    hipMalloc(&d_output, output_size * sizeof(float));

    hipMemcpy(d_input, input, input_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel, kernel_size * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, output_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);
}

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    std::ifstream input_file;
    std::string filename = argv[1];
    
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }

    int input_size, kernel_size, output_size;
    input_file >> input_size >> kernel_size;

    output_size = input_size - kernel_size + 1;

    std::vector<float> input(input_size), kernel(kernel_size), output(output_size);
    
    for (int i = 0; i < input_size; ++i){
        input_file >> input[i];
    }
    for (int i = 0; i < kernel_size; ++i){
        input_file >> kernel[i];
    }

    input_file.close();

    solve(input.data(), kernel.data(), output.data(), input_size, kernel_size);

    for(int i = 0; i < output_size; ++i){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}