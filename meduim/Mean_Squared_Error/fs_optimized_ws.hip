#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

#define BLOCK_DIM     256       
#define COARSE_FACTOR 4         

__device__ __forceinline__ float Square(float x) { return x * x; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    const int W = warpSize;
#if defined(__HIP_PLATFORM_NVIDIA__)
    unsigned m = __activemask();
    for (int off = W >> 1; off > 0; off >>= 1)
        v += __shfl_down_sync(m, v, off, W);
#else
    for (int off = W >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off, W);
#endif
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    const int W = warpSize;
    int lane = threadIdx.x % W;
    int wid  = threadIdx.x / W;

    v = warp_reduce_sum(v);

    if (blockDim.x <= W) {
        return (lane == 0) ? v : 0.0f;
    }

    extern __shared__ float sm[];                       
    if (lane == 0) sm[wid] = v;
    __syncthreads();

    float agg = 0.0f;
    if (wid == 0) {
        int numWarps = (blockDim.x + W - 1) / W;        
        agg = (lane < numWarps) ? sm[lane] : 0.0f;
        agg = warp_reduce_sum(agg);
        return (lane == 0) ? agg : 0.0f;
    }
    return 0.0f;
}

__global__ void MseKernel_WarpShuffle(const float* __restrict__ Pred,
                                      const float* __restrict__ Target,
                                      float* __restrict__ Mse,
                                      int N, float invN)
{
    int tid  = COARSE_FACTOR * BLOCK_DIM * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int pos = tid + i * BLOCK_DIM;
        if (pos < N) {
            float d = Pred[pos] - Target[pos];
            sum += d * d;                 
        }
    }

    // block reduce（只有 thread 0 會拿到非 0 的 blockSum）
    float blockSum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(Mse, blockSum * invN);
    }
}

void solve(const float *predictions, const float *targets, float *mse, int N)
{
    float *d_predictions = nullptr, *d_targets = nullptr, *d_mse = nullptr;

    hipMalloc(&d_predictions, N * sizeof(float));
    hipMalloc(&d_targets,     N * sizeof(float));
    hipMalloc(&d_mse,         sizeof(float));

    hipMemcpy(d_predictions, predictions, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_targets,     targets,     N * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_mse, 0, sizeof(float));

    const int block = BLOCK_DIM;
    const int grid  = (N + block * COARSE_FACTOR - 1) / (block * COARSE_FACTOR);

    const int numWarps = (block + warpSize - 1) / warpSize;
    const size_t shmem_bytes = sizeof(float) * numWarps;

    float invN = (N > 0) ? (1.0f / static_cast<float>(N)) : 0.0f;

    hipLaunchKernelGGL(
        MseKernel_WarpShuffle,
        dim3(grid), dim3(block),
        shmem_bytes, 0,
        d_predictions, d_targets, d_mse, N, invN
    );
    hipDeviceSynchronize();

    hipMemcpy(mse, d_mse, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_predictions);
    hipFree(d_targets);
    hipFree(d_mse);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error " << argv[1] << std::endl;
        return 1;
    }

    int N;
    input_file >> N;

    std::vector<float> predictions(N), targets(N);
    for (int i = 0; i < N; ++i) input_file >> predictions[i];
    for (int i = 0; i < N; ++i) input_file >> targets[i];
    input_file.close();

    float mse = 0.0f;
    solve(predictions.data(), targets.data(), &mse, N);
    std::cout << mse << std::endl;
    return 0;
}
