#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <fstream>

__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    atomicAdd(output, input[idx]); 
}

extern "C" void solve(const float* input, float* output, int N) {  
    float *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(float));
    
    int threadperblock = 256;
    int blockpergrid = (N + threadperblock - 1) / threadperblock;

    reduction<<<blockpergrid, threadperblock>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, sizeof(float), hipMemcpyDeviceToHost);

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
    int N;
    float output;
    
    input_file >> N;
    std::vector<float> input(N);

    for(int i = 0; i < N; ++i) {
        input_file >> input[i];
    }

    input_file.close();

    solve(input.data(), &output, N);

    std::cout << output << std::endl;
    
    return 0;
}