#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    //#pragma unroll 4 //should be faster
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= N) return;

    output[idx] = input[idx];
    for(int i = 0; i < R; ++i) output[idx] = fnv1a_hash(output[idx]);
}

void solve(const int* input, unsigned int* output, int N, int R) {

    int *d_input;
    unsigned int *d_output;

    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(unsigned int));

    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, R);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, N * sizeof(unsigned int), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
}

int main(){
    int N, R;
    std::cin >> N >> R;

    std::vector<int> input(N);
    std::vector<unsigned int> output(N);

    for(int i = 0; i < N; ++i) std::cin >> input[i];

    solve(input.data(), output.data(), N, R);

    for(int i = 0; i < N; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
}