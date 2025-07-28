#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <float.h>

#include <fstream>

#define threadperblock 1024

__device__ void atomicMaxfloat(float *const addr, const float val) {
     if (*addr >= val) return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

// findmax優化後反而比較慢(一點點)，可能是bank conflict? 破thread沒return? 試試看warpshuffle?
__global__ void findmax(const float* input, float* globalmax, int N) {
    __shared__ float sdata[threadperblock];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    sdata[tidx] = (idx < N) ? input[idx] : 0.0f;
    
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        // bank conflict?
        if (tidx < i) sdata[tidx] = fmax(sdata[tidx], sdata[tidx + i]);
        __syncthreads();
    }

    if (tidx == 0) atomicMaxfloat(globalmax, sdata[0]); 
}

// 快了兩倍多
__global__ void exponentialsum(const float* input, float* output, int N, float globalmax, float* globalsum) {
    __shared__ float sdata[threadperblock];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    float val = expf(input[idx] - globalmax);
    output[idx] = val;

    sdata[tidx] = (idx < N) ? val : 0.0f;
    
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }
    
    if (tidx == 0) atomicAdd(globalsum, sdata[0]); 
}

__global__ void softmax(const float* input, float* output, int N, float globalsum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return; 
    
    output[idx] /= globalsum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {

    int blocksPerGrid = (N + threadperblock - 1) / threadperblock;

    float *d_input, *d_output, *d_globalmax, *d_globalsum;
    float globalmax, globalsum;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_globalmax, sizeof(float));
    hipMalloc(&d_globalsum, sizeof(float));

    float init_max = -FLT_MAX;
    float init_sum = 0.0;
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalmax, &init_max, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalsum, &init_sum, sizeof(float), hipMemcpyHostToDevice);

    findmax<<<blocksPerGrid, threadperblock>>>(d_input, d_globalmax, N);
    hipMemcpy(&globalmax, d_globalmax, sizeof(float), hipMemcpyDeviceToHost);

    exponentialsum<<<blocksPerGrid, threadperblock>>>(d_input, d_output, N, globalmax, d_globalsum);
    hipMemcpy(&globalsum, d_globalsum, sizeof(float), hipMemcpyDeviceToHost);

    softmax<<<blocksPerGrid, threadperblock>>>(d_input, d_output, N, globalsum);
    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_globalmax);
    hipFree(d_globalsum);
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

    std::vector<float> input(N), output(N);

    for(int i = 0; i < N; ++i) input_file >> input[i];

    input_file.close();

    solve(input.data(), output.data(), N);

    for(int i = 0; i < N; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
}