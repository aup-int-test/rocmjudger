#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

/*sample input
4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5

0.25

- - -
*/

#define BLOCK_DIM (1024)
#define COARSE_FACTOR 32 // element deal per thread

__device__ __forceinline__ float Square(float x) { return x * x; }

__global__ void MseKernel_NoShuffle(const float *Pred, const float *Target, float *Mse, int N)
{
    __shared__ float sdata[BLOCK_DIM];

    int tid  = threadIdx.x;
    int base = COARSE_FACTOR * BLOCK_DIM * blockIdx.x + tid;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int pos = base + i * BLOCK_DIM;
        if (pos < N) {
            float d = Pred[pos] - Target[pos];
            sum += Square(d) / N;   
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = BLOCK_DIM >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(Mse, sdata[0]);
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

    int grid = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);
    MseKernel_NoShuffle<<<grid, BLOCK_DIM>>>(d_predictions, d_targets, d_mse, N);
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
    std::cout <<  mse << std::endl;
    return 0;
}
