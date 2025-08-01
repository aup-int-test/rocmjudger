#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

/*
2 3
1 2
1 2 3
4 5 6
1 0
*/
__constant__ int c_kernel[64 * 64];

__global__ void convolution2D_kernel(const int* input, int* output, 
                                  int input_rows, int input_cols, 
                                  int kernel_rows, int kernel_cols,
                                  int output_rows, int output_cols) {
    
    int tile_rows = blockDim.y + kernel_rows - 1;
    int tile_cols = blockDim.x + kernel_cols - 1;
    
    __shared__ int s_input[8132];
    
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tile_start_row = blockIdx.y * blockDim.y;
    int tile_start_col = blockIdx.x * blockDim.x;
    
    for (int i = local_row; i < tile_rows; i += blockDim.y) {
        for (int j = local_col; j < tile_cols; j += blockDim.x) {
            int input_row = tile_start_row + i;
            int input_col = tile_start_col + j;
            
            if (input_row < input_rows && input_col < input_cols) {
                s_input[i * tile_cols + j] = input[input_row * input_cols + input_col];
            } else {
                s_input[i * tile_cols + j] = 0;  // padding
            }
        }
    }
    
    __syncthreads();
    
    if (global_row < output_rows && global_col < output_cols) {
        int sum = 0;
        
        for (int kr = 0; kr < kernel_rows; kr++) {
            for (int kc = 0; kc < kernel_cols; kc++) {
                int shared_row = local_row + kr;
                int shared_col = local_col + kc;
                sum += s_input[shared_row * tile_cols + shared_col] * 
                       c_kernel[kr * kernel_cols + kc];
            }
        }
        
        output[global_row * output_cols + global_col] = sum;
    }
}

void solve(const int* input, const int* kernel, int* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols){
        
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t inputsize = input_rows * input_cols * sizeof(int);
    size_t kernelsize = kernel_rows * kernel_cols * sizeof(int);
    size_t outputsize = output_rows * output_cols * sizeof(int);

    int *d_input, *d_kernel, *d_output;

    hipMalloc(&d_input, inputsize);
    hipMalloc(&d_kernel, kernelsize);
    hipMalloc(&d_output, outputsize);

    hipMemcpy(d_input, input, inputsize, hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel, kernelsize, hipMemcpyHostToDevice);
    //hipMemset(d_output, 0, outputsize);
    hipMemcpyToSymbol(c_kernel, kernel, kernelsize);

    dim3 threads(16, 16);  
    dim3 blocks((output_cols + threads.x - 1) / threads.x, (output_rows + threads.y - 1) / threads.y);

    convolution2D_kernel<<<blocks, threads>>>(
            d_input, d_output, input_rows, input_cols, 
            kernel_rows, kernel_cols, output_rows, output_cols);
    
    hipDeviceSynchronize();
    hipMemcpy(output, d_output, outputsize, hipMemcpyDeviceToHost);
    
    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);
}

int main(int argc, char* argv[]) {
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
    int input_rows, input_cols, kernel_rows, kernel_cols;
    input_file >> input_rows >> input_cols >> kernel_rows >> kernel_cols;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    std::vector<int> input(input_rows * input_cols), kernel(kernel_rows * kernel_cols), output(output_rows * output_cols);

    for(int i = 0; i < input_rows; ++i) for(int j = 0; j < input_cols; ++j) input_file >> input[i * input_cols + j];
    for(int i = 0; i < kernel_rows; ++i) for(int j = 0; j < kernel_cols; ++j) input_file >> kernel[i * kernel_cols + j];

    input_file.close();

    solve(input.data(), kernel.data(), output.data(), input_rows, input_cols, kernel_rows, kernel_cols);

    for(int i = 0; i < output_rows; ++i){
        for(int j = 0; j < output_cols; ++j) std::cout << output[i * output_cols + j] << " ";
        std::cout << std::endl;
    }
}