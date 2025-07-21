#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= N * N) return;

    B[idx] = A[idx];
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {

    float *d_A, *d_B;
    int total = N * N;

    hipMalloc(&d_A, total * sizeof(float));
    hipMalloc(&d_B, total * sizeof(float));

    hipMemcpy(d_A, A, total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, total * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    hipDeviceSynchronize();
} 

int main(){
    int N;
    std::cin >> N;

    std::vector<float> A(N * N), B(N * N);

    for(int i = 0; i < N; ++i) for(int j = 0; j < N; ++j) std::cin >> A[i * N + j];

    solve(A.data(), B.data(), N);

    //std::cout << std::fixed << std::setprecision(3);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            std::cout << A[i * N + j] << " ";
        } 
        std::cout << std::endl;
    }
}