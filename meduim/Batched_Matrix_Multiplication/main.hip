#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void mmAB(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int batch_id = blockIdx.z;
    A += batch_id * M * K;
    B += batch_id * N * K;
    C += batch_id * M * N;
    if (row < M && col < N) {
        float rC = 0.f;
        for (int k = 0; k < K; ++k) rC += A[row * K + k] * B[k * N + col];
        C[row * N + col] = rC;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       BATCH);
    hipLaunchKernelGGL(mmAB, blocksPerGrid, threadsPerBlock, 0, 0, A, B, C, BATCH, M, N, K);
}

int main() {
    int BATCH, M, N, K;
    cin >> BATCH >> M >> N >> K;
    vector<float> hA(BATCH * M * K);
    vector<float> hB(BATCH * K * N);
    vector<float> hC(BATCH * M * N);
    for (int i = 0; i < BATCH * M * K; ++i) cin >> hA[i];
    for (int i = 0; i < BATCH * K * N; ++i) cin >> hB[i];
    float *dA, *dB, *dC;
    hipMalloc(&dA, hA.size() * sizeof(float));
    hipMalloc(&dB, hB.size() * sizeof(float));
    hipMalloc(&dC, hC.size() * sizeof(float));
    hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice);
    solve(dA, dB, dC, BATCH, M, N, K);
    hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost);
    for (int b = 0; b < BATCH; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j)
                cout << hC[b * M * N + i * N + j] << (j + 1 == N ? '\n' : ' ');
        }
    }
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    return 0;
}
