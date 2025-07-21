#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

#define threadperblock 256

__global__ void reduction(const float* input, float* output, int N){
    __shared__ float sdata[threadperblock];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    sdata[tidx] = (idx < N) ? input[idx] : 0.0f;

    __syncthreads();

    // Parallel reduction within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }

    if (tidx == 0) atomicAdd(output, sdata[0]); 
}


extern "C" void solve(const float* input, float* output, int N) {  

    float *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    int blockpergrid = (N + threadperblock - 1) / threadperblock;

    //size_t shared_mem_size = threadperblock * sizeof(float);
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