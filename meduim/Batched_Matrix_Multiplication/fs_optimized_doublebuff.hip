#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void mmAB_fast(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int BATCH, int M, int N, int K)
{
    int batch_id = blockIdx.z;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    A += batch_id * M * K;
    B += batch_id * K * N;
    C += batch_id * M * N;
    extern __shared__ float smem[];
    float* As0 = smem;
    float* Bs0 = smem + BM * BK;
    float* As1 = Bs0 + BK * BN;
    float* Bs1 = As1 + BM * BK;
    const float* Ablock = A + blockRow * BM * K;
    const float* Bblock = B + blockCol * BN;
    float* Cblock = C + blockRow * BM * N + blockCol * BN;
    float cReg[TM][TN] = {0.0f};
    int tid = threadRow * blockDim.x + threadCol;
    const int numThreads = blockDim.x * blockDim.y;
    int curr = 0;
    for (int i = tid; i < BM * BK; i += numThreads) {
        int row = i / BK;
        int col = i % BK;
        if (blockRow * BM + row < M && col < K)
            As0[i] = Ablock[row * K + col];
        else
            As0[i] = 0.0f;
    }
    for (int i = tid; i < BK * BN; i += numThreads) {
        int row = i / BN;
        int col = i % BN;
        if (row < K && blockCol * BN + col < N)
            Bs0[i] = Bblock[row * N + col];
        else
            Bs0[i] = 0.0f;
    }
    __syncthreads();
    for (int bk = 0; bk < K; bk += BK) {
        if (bk + BK < K) {
            if (curr == 0) {
                for (int i = tid; i < BM * BK; i += numThreads) {
                    int row = i / BK;
                    int col = i % BK;
                    if (blockRow * BM + row < M && bk + BK + col < K)
                        As1[i] = Ablock[row * K + (bk + BK + col)];
                    else
                        As1[i] = 0.0f;
                }
                for (int i = tid; i < BK * BN; i += numThreads) {
                    int row = i / BN;
                    int col = i % BN;
                    if (bk + BK + row < K && blockCol * BN + col < N)
                        Bs1[i] = Bblock[(bk + BK + row) * N + col];
                    else
                        Bs1[i] = 0.0f;
                }
            } else {
                for (int i = tid; i < BM * BK; i += numThreads) {
                    int row = i / BK;
                    int col = i % BK;
                    if (blockRow * BM + row < M && bk + BK + col < K)
                        As0[i] = Ablock[row * K + (bk + BK + col)];
                    else
                        As0[i] = 0.0f;
                }
                for (int i = tid; i < BK * BN; i += numThreads) {
                    int row = i / BN;
                    int col = i % BN;
                    if (bk + BK + row < K && blockCol * BN + col < N)
                        Bs0[i] = Bblock[(bk + BK + row) * N + col];
                    else
                        Bs0[i] = 0.0f;
                }
            }
        }
        float* As = curr ? As1 : As0;
        float* Bs = curr ? Bs1 : Bs0;
        for (int kk = 0; kk < BK; ++kk) {
            float aReg[TM];
            float bReg[TN];
            int aBase = (threadRow * TM) * BK + kk;
            int bBase = kk * BN + threadCol * TN;
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                aReg[i] = As[aBase + i * BK];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bReg[j] = Bs[bBase + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    cReg[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
        curr ^= 1;
    }
    int rowBase = blockRow * BM + threadRow * TM;
    int colBase = blockCol * BN + threadCol * TN;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        if (rowBase + i < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                if (colBase + j < N) {
                    Cblock[(threadRow * TM + i) * N + (threadCol * TN + j)] = cReg[i][j];
                }
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    size_t smem = 2 * (BM * BK + BK * BN) * sizeof(float);
    hipLaunchKernelGGL(mmAB_fast, grid, block, smem, 0, A, B, C, BATCH, M, N, K);
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
