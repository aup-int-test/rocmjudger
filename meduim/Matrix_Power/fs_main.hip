#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#define TILE 16

// =============================
// GPU Kernel: Naive MatMul
// =============================
__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;
}

// =============================
// Host Function: Matrix Power
// =============================
void solve(const float* d_input, float* d_output, int N, int P) {
    if (P == 0) {
        std::vector<float> identity(N * N, 0.0f);
        for (int i = 0; i < N; ++i)
            identity[i * N + i] = 1.0f;
        hipMemcpy(d_output, identity.data(), N * N * sizeof(float), hipMemcpyHostToDevice);
        return;
    }

    float *A, *C;
    hipMalloc(&A, N * N * sizeof(float));
    hipMalloc(&C, N * N * sizeof(float));

    hipMemcpy(A, d_input, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Initialize output = input
    hipMemcpy(d_output, d_input, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    for (int p = 1; p < P; ++p) {
        hipLaunchKernelGGL(matmul, blocks, threads, 0, 0, A, d_input, C, N);
        hipDeviceSynchronize();
        hipMemcpy(A, C, N * N * sizeof(float), hipMemcpyDeviceToDevice);
    }

    hipMemcpy(d_output, A, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    hipFree(A);
    hipFree(C);
}

// =============================
// Main Function: fstream I/O
// =============================
int main() {
    std::ifstream fin("input.in");
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open input.in" << std::endl;
        return 1;
    }

    int N, P;
    fin >> N >> P;

    std::vector<float> h_input(N * N);
    for (int i = 0; i < N * N; ++i)
        fin >> h_input[i];
    fin.close();

    float *d_input, *d_output;
    hipMalloc(&d_input, N * N * sizeof(float));
    hipMalloc(&d_output, N * N * sizeof(float));

    hipMemcpy(d_input, h_input.data(), N * N * sizeof(float), hipMemcpyHostToDevice);

    solve(d_input, d_output, N, P);

    std::vector<float> h_output(N * N);
    hipMemcpy(h_output.data(), d_output, N * N * sizeof(float), hipMemcpyDeviceToHost);

    // Output to stdout (cout)
    for (int i = 0; i < N * N; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_output[i];
        if ((i + 1) % N == 0)
            std::cout << "\n";
        else
            std::cout << " ";
    }

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
