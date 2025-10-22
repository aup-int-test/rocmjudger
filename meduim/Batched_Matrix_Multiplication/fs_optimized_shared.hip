#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define TILE 16

__global__ void mmAB_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int BATCH, int M, int N, int K)
{
    int batch_id = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    const float* Abatch = A + batch_id * M * K;
    const float* Bbatch = B + batch_id * K * N;
    float* Cbatch = C + batch_id * M * N;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? Abatch[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? Bbatch[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        Cbatch[row * N + col] = sum;
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1) / TILE,
                       (M + TILE - 1) / TILE,
                       BATCH);
    hipLaunchKernelGGL(mmAB_tiled, blocksPerGrid, threadsPerBlock, 0, 0, A, B, C, BATCH, M, N, K);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    ifstream fin(argv[1]);
    int BATCH, M, N, K;
    fin >> BATCH >> M >> N >> K;
    vector<float> hA(BATCH * M * K);
    vector<float> hB(BATCH * K * N);
    vector<float> hC(BATCH * M * N);
    for (int i = 0; i < BATCH * M * K; ++i) fin >> hA[i];
    for (int i = 0; i < BATCH * K * N; ++i) fin >> hB[i];
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
