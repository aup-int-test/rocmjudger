#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

/*sample input
4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5

0.25

- - -


*/

#define BLOCK_DIM (256)
#define COARSE_FACTOR 4
#define WARP_SIZE 32

__device__ float Square(float X)
{
    return X * X;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    static __shared__ float shared[WARP_SIZE]; 
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val); 

    if (lane == 0)
        shared[wid] = val; 

    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val); 

    return val;
}

__global__ void MseKernel(const float *Pred, const float *Target, float *Mse, int N)
{
    int Tid = COARSE_FACTOR * BLOCK_DIM * blockIdx.x + threadIdx.x;

    float Sum = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        int Pos = Tid + BLOCK_DIM * I;
        if (Pos < N)
        {
            Sum += Square(Pred[Pos] - Target[Pos]) / N;
        }
    }

    Sum = blockReduceSum(Sum);

    if (threadIdx.x == 0)
    {
        atomicAdd(Mse, Sum);
    }
}

void solve(const float *predictions, const float *targets, float *mse, int N)
{
    float *d_predictions, *d_targets, *d_mse;

    hipMalloc(&d_predictions, N * sizeof(float));
    hipMalloc(&d_targets, N * sizeof(float));
    hipMalloc(&d_mse, sizeof(float));

    hipMemcpy(d_predictions, predictions, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_targets, targets, N * sizeof(float), hipMemcpyHostToDevice);

    int GridDim = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);
    MseKernel<<<GridDim, BLOCK_DIM>>>(d_predictions, d_targets, d_mse, N);
    hipDeviceSynchronize();

    hipMemcpy(mse, d_mse, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_predictions);
    hipFree(d_targets);
    hipFree(d_mse);
}

int main(){

    int N;
    std::cin >> N;

    std::vector<float> predictions(N), targets(N);

    for(int i = 0; i < N; ++i) std::cin >> predictions[i];
    for(int i = 0; i < N; ++i) std::cin >> targets[i];

    float mse;

    solve(predictions.data(), targets.data(), &mse, N);

    std::cout << mse << std::endl;
}