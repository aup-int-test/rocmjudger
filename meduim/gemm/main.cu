#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define threadperblock 256

__global__ void kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    
    if (row < M && col < N) {

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        
        half old_c = C[row * N + col];
        C[row * N + col] = __hadd(__float2half(alpha * sum), __float2half(beta * __half2float(old_c)));
    }
}

void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

    half *d_A, *d_B, *d_C;

    hipMalloc(&d_A, M * K * sizeof(half));
    hipMalloc(&d_B, K * N * sizeof(half));
    hipMalloc(&d_C, M * N * sizeof(half));

    hipMemcpy(d_A, A, M * K * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, K * N* sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_C, C, M * N* sizeof(half), hipMemcpyHostToDevice);

    int blocks = (M * N + threadperblock - 1) / threadperblock;

    kernel<<<blocks, threadperblock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    hipMemcpy(C, d_C, M * N * sizeof(half), hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main(){

    int M, N, K;
    float alpha, beta;
    
    std::cin >> M >> N >> K;
    std::cin >> alpha >> beta;

    std::vector<half> A(M * K), B(K * N), C(M * N);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
            float temp;
            std::cin >> temp;
            A[i * K + j] = __float2half(temp);
        }
    }

    for(int i = 0; i < K; ++i){
        for(int j = 0; j < N; ++j){
            float temp;
            std::cin >> temp;
            B[i * N + j] = __float2half(temp);
        }
    }

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float temp;
            std::cin >> temp;
            C[i * N + j] = __float2half(temp);
        }
    }

    solve(A.data(), B.data(), C.data(), M, N, K, alpha, beta);

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float temp;
            C[i * N + j] = __float2half(temp);
            std::cout << temp;
        }
    }
}