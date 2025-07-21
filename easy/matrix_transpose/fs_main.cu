#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < cols && idy < rows){
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

void solve(const float* input, float* output, int rows, int cols) {

    float *d_input, *d_output;

    hipMalloc(&d_input, rows * cols * sizeof(float));
    hipMalloc(&d_output, rows * cols * sizeof(float));

    hipMemcpy(d_input, input, rows * cols * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, rows * cols * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
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
    int rows, cols;
    input_file >> rows >> cols;

    std::vector<float> h_input(rows * cols), h_output(rows * cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input_file >> h_input[i * cols + j];
        }
    }

    input_file.close();

    solve(h_input.data(), h_output.data(), rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_output[i * cols + j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}