#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= N * N) return;

    B[idx] = A[idx];
}

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
    int N;
    input_file >> N;

    std::vector<float> A(N * N), B(N * N);

    for(int i = 0; i < N; ++i) for(int j = 0; j < N; ++j) input_file >> A[i * N + j];

    input_file.close();

    solve(A.data(), B.data(), N);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            std::cout << A[i * N + j] << " ";
        } 
        std::cout << std::endl;
    }
}