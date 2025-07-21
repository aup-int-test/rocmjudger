#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int r_idx = N - idx - 1;

    if(idx < r_idx){
        float tmp = input[idx];
        input[idx] = input[r_idx];
        input[r_idx] = tmp;
    }
}

// input is device pointer
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

int main(){
    int N;
    std::cin >> N;

    std::vector<float> input(N);

    for(int i = 0; i < N; ++i){
        std::cin >> input[i];
    }

    solve(input.data(), N);

    // Output the resulting vector
    for(int i = 0; i < N; ++i){
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}