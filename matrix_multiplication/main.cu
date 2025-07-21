#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < K){
        float sum = 0.0f;
        
        for(int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory
    hipMalloc(&d_A, M * N * sizeof(float));
    hipMalloc(&d_B, N * K * sizeof(float));
    hipMalloc(&d_C, M * K * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_A, A, M * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N * K * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    hipDeviceSynchronize();

    hipMemcpy(C, d_C, M * K * sizeof(float), hipMemcpyDeviceToHost);

    // Free GPU memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main(){
    int M, N, K;
    std::cin >> M >> N >> K;

    // Read input vectors from standard input
    std::vector<float> h_A(M * N), h_B(N * K), h_C(M * K);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cin >> h_A[i * N + j];
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cin >> h_B[i * K + j];
        }
    }

    // Call the solve function
    solve(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    std::cout << std::fixed << std::setprecision(3);

    // Output the resulting vector
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << std::setw(8) << h_C[i * K + j];
            if (j < K - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}