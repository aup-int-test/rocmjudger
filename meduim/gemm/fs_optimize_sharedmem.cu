#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>

#include <fstream>

#define TILE_SIZE 16

__global__ void kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta){
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Matrix multiplication: sum = A * B
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = __float2half(0.0f);
            
        if (t * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = __float2half(0.0f);
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Final computation: C = alpha * A * B + beta * C
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = __hadd(__float2half(alpha * sum), __float2half(beta * __half2float(C[idx])));
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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    hipMemcpy(C, d_C, M * N * sizeof(half), hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
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

    int M, N, K;
    float alpha, beta;
    
    input_file >> M >> N >> K;
    input_file >> alpha >> beta;

    std::vector<half> A(M * K), B(K * N), C(M * N);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
            float temp;
            input_file >> temp;
            A[i * K + j] = __float2half(temp);
        }
    }

    for(int i = 0; i < K; ++i){
        for(int j = 0; j < N; ++j){
            float temp;
            input_file >> temp;
            B[i * N + j] = __float2half(temp);
        }
    }

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float temp;
            input_file >> temp;
            C[i * N + j] = __float2half(temp);
        }
    }

    input_file.close();

    solve(A.data(), B.data(), C.data(), M, N, K, alpha, beta);

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float temp;
            C[i * N + j] = __float2half(temp);
            std::cout << temp;
        }
    }
}