#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

__global__ void reverse_array(float* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int r_idx = N - idx - 1;

    if(idx < r_idx){
        float tmp = input[idx];
        input[idx] = input[r_idx];
        input[r_idx] = tmp;
    }
}

void solve(float* input, int N) {

    float *d_input;

    hipMalloc(&d_input, N * sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    hipDeviceSynchronize();

    hipMemcpy(input, d_input, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
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
    int N;
    input_file >> N;

    std::vector<float> input(N);

    for(int i = 0; i < N; ++i){
        input_file >> input[i];
    }

    input_file.close();

    solve(input.data(), N);

    for(int i = 0; i < N; ++i){
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}