#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < cols && idy < rows){
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {

    float *d_input, *d_output;

    hipMalloc(&d_input, rows * cols * sizeof(float));
    hipMalloc(&d_output, rows * cols * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_input, input, rows * cols * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, rows * cols * sizeof(float), hipMemcpyDeviceToHost);

    // Free GPU memory
    hipFree(d_input);
    hipFree(d_output);
}

int main(){
    int rows, cols;
    std::cin >> rows >> cols;

    // Read input vectors from standard input
    std::vector<float> h_input(rows * cols), h_output(rows * cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cin >> h_input[i * cols + j];
        }
    }

    // Call the solve function
    solve(h_input.data(), h_output.data(), rows, cols);

    std::cout << std::fixed << std::setprecision(3);

    // Output the resulting vector
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << h_output[i * cols + j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}