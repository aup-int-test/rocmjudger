#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#define TILE 16

// =====================================================
// Shared-memory tiled matrix multiplication
// =====================================================
__global__ void matmul_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// =====================================================
// Utility: Initialize matrix as identity
// =====================================================
__global__ void set_identity(float* M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N)
        M[idx] = (idx / N == idx % N) ? 1.0f : 0.0f;
}

// =====================================================
// Utility: Copy matrix A → B
// =====================================================
__global__ void copy_matrix(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N)
        B[idx] = A[idx];
}

// =====================================================
// Host Function: Matrix Power
// =====================================================
void solve(const float* d_input, float* d_output, int N, int P) {
    if (P == 0) {
        int threads = 256;
        int blocks = (N * N + threads - 1) / threads;
        hipLaunchKernelGGL(set_identity, dim3(blocks), dim3(threads), 0, 0, d_output, N);
        hipDeviceSynchronize();
        return;
    }

    float *A, *B, *C;
    hipMalloc(&A, N * N * sizeof(float));
    hipMalloc(&B, N * N * sizeof(float));
    hipMalloc(&C, N * N * sizeof(float));

    hipMemcpy(A, d_input, N * N * sizeof(float), hipMemcpyDeviceToDevice);
    hipMemcpy(B, d_input, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    hipMemcpy(d_output, d_input, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    for (int p = 1; p < P; ++p) {
        hipLaunchKernelGGL(matmul_shared, blocks, threads, 0, 0, A, d_input, C, N);
        hipDeviceSynchronize();
        hipMemcpy(A, C, N * N * sizeof(float), hipMemcpyDeviceToDevice);
    }

    hipMemcpy(d_output, A, N * N * sizeof(float), hipMemcpyDeviceToDevice);

    hipFree(A);
    hipFree(B);
    hipFree(C);
}

// =====================================================
// Main Function: fstream input + cout output
// =====================================================
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
