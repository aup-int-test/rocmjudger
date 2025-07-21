/*broken*/
#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

/* basic testcase
0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
*/

__inline__ __device__ float warpReduceSum(float val){
    // full mask for 32 threads
    for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down(val, offset);

    return val;
}

__global__ void montecarlo(const float* y_samples, float* result, float a, float b, int n_samples){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    float sum = 0.0f;
    if(idx < n_samples) sum = (b - a) * y_samples[idx] / n_samples;
    
    sum = warpReduceSum(sum);

    // reduction in a single block
    __shared__ float blockSum;
    if((tidx % 32) == 0) {
        blockSum = 0;
        atomicAdd(&blockSum, sum);  
    }

    __syncthreads();

    // all reduction
    if(tidx == 0) atomicAdd(result, blockSum);
}

// y_samples, result are device pointers
void solve(const float* y_samples, float* result, float a, float b, int n_samples) {

    float *d_ysamples, *d_result;

    hipMalloc(&d_ysamples, n_samples * sizeof(float));
    hipMalloc(&d_result, sizeof(float));

    hipMemcpy(d_ysamples, y_samples, n_samples * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_result, 0, sizeof(float));

    int threadperblock = 256;
    int blockpergrid = (n_samples + threadperblock - 1) / threadperblock;

    montecarlo<<<blockpergrid, threadperblock>>>(d_ysamples, d_result, a, b, n_samples);
    hipDeviceSynchronize();

    hipMemcpy(result, d_result, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_ysamples);
    hipFree(d_result);
}

int main(){
    int a, b, n_samples;
    float result;
    std::cin >> a >> b >> n_samples;

    std::vector<float> y_samples(n_samples);

    for(int i = 0; i < n_samples; ++i) std::cin >> y_samples[i];

    solve(y_samples.data(), &result, a, b, n_samples);

    std::cout << result << std::endl;
}
