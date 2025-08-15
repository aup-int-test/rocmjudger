#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

__global__ void reduction(const int* input, int* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    atomicAdd(output, input[idx]); 
}


extern "C" void solve(const int* input, int* output, int N) {  

    int *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, sizeof(int));

    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(int));
    
    int threadperblock = 256;
    int blockpergrid = (N + threadperblock - 1) / threadperblock;

    reduction<<<threadperblock, blockpergrid>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
}

int main(){
    int N;
    int output;
    std::cin >> N;

    std::vector<int> input(N);

    for(int i = 0; i < N; ++i) std::cin >> input[i];

    solve(input.data(), &output, N);

    std::cout << output << std::endl;
}