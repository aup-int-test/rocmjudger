#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#define threadperblock 256

__global__ void reduction(const int* input, int* output, int N){
    __shared__ int sdata[threadperblock];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    sdata[tidx] = (idx < N) ? input[idx] : 0.0f;

    // Parallel reduction within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }

    if (tidx == 0) atomicAdd(output, sdata[0]); 
}


extern "C" void solve(const int* input, int* output, int N) {  

    int *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, sizeof(int));

    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(int));
    
    int blockpergrid = (N + threadperblock - 1) / threadperblock;

    //size_t shared_mem_size = threadperblock * sizeof(int);
    reduction<<<blockpergrid, threadperblock>>>(d_input, d_output, N);
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