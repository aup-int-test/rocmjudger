#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N){
        output[idx] = max(input[idx], 0.0f);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {

    float *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
}

int main(){
    int N;
    std::cin >> N;

    std::vector<float> input(N), output(N);

    for(int i = 0; i < N; ++i) std::cin >> input[i];

    solve(input.data(), output.data(), N);

    for(int i = 0; i < N; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
}
