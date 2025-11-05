#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N) C[id] = A[id] + B[id];
}

void solve(const float* A, const float* B, float* C, int N) {

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    hipDeviceSynchronize();

    hipMemcpy(C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main() {
    int N;
    std::cin >> N;

    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (int i = 0; i < N; ++i) {
        std::cin >> h_A[i];
    }
    for (int i = 0; i < N; ++i) {
        std::cin >> h_B[i];
    }

    solve(h_A.data(), h_B.data(), h_C.data(), N);

    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << (i + 1 < N ? ' ' : '\n');
    }

    return 0;
}