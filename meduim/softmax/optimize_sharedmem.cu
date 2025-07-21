#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <float.h>

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

__global__ void findmax(const float* input, float* globalmax, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    atomicMaxfloat(globalmax, input[idx]); 
}

__global__ void exponentialsum(const float* input, float* output, int N, float globalmax, float* globalsum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    float val = expf(input[idx] - globalmax);
    output[idx] = val;

    atomicAdd(globalsum, val); 
}

__global__ void softmax(const float* input, float* output, int N, float globalsum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return; 

    output[idx] /= globalsum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

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

    findmax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_globalmax, N);
    hipMemcpy(&globalmax, d_globalmax, sizeof(float), hipMemcpyDeviceToHost);

    exponentialsum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, globalmax, d_globalsum);
    hipMemcpy(&globalsum, d_globalsum, sizeof(float), hipMemcpyDeviceToHost);

    softmax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, globalsum);
    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_globalmax);
    hipFree(d_globalsum);
}

int main(){
    int N;
    std::cin >> N;

    std::vector<float> input(N), output(N);

    for(int i = 0; i < N; ++i) std::cin >> input[i];

    solve(input.data(), output.data(), N);

    for(int i = 0; i < N; ++i) std::cout << std::fixed << std::setprecision(3) << output[i] << " ";
    std::cout << std::endl;
}