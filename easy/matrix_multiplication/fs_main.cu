#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

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

void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, M * N * sizeof(float));
    hipMalloc(&d_B, N * K * sizeof(float));
    hipMalloc(&d_C, M * K * sizeof(float));

    hipMemcpy(d_A, A, M * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N * K * sizeof(float), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    hipDeviceSynchronize();

    hipMemcpy(C, d_C, M * K * sizeof(float), hipMemcpyDeviceToHost);

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
    input_file >> M >> N >> K;

    std::vector<float> h_A(M * N), h_B(N * K), h_C(M * K);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input_file >> h_A[i * N + j];
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            input_file >> h_B[i * K + j];
        }
    }

    input_file.close();

    solve(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << std::setw(8) << h_C[i * K + j];
            if (j < K - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}