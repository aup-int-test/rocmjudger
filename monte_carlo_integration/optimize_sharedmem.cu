#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <fstream>

/* basic testcase
0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
*/

__global__ void montecarlo(const double* y_samples, double* result, double a, double b, int n_samples) {
    extern __shared__ double sdata[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    if (idx < n_samples)
        sdata[tidx] = (b - a) * y_samples[idx] / n_samples;
    else
        sdata[tidx] = 0.0;
    __syncthreads();

    // Parallel reduction within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }

    if (tidx == 0) atomicAdd(result, sdata[0]); // atomicAdd for double is supported on recent hardware
}

void solve(const double* y_samples, double* result, double a, double b, int n_samples) {
    double* d_ysamples;
    double* d_result;

    hipMalloc(&d_ysamples, n_samples * sizeof(double));
    hipMalloc(&d_result, sizeof(double));

    hipMemcpy(d_ysamples, y_samples, n_samples * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_result, 0, sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;

    size_t shared_mem_size = threadsPerBlock * sizeof(double);
    montecarlo<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_ysamples, d_result, a, b, n_samples);
    hipDeviceSynchronize();

    hipMemcpy(result, d_result, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_ysamples);
    hipFree(d_result);
}

int main() {
    int n_samples;
    double a, b, result;
    std::cin >> a >> b >> n_samples;

    std::vector<double> y_samples(n_samples);

    for(int i = 0; i < n_samples; ++i) std::cin >> y_samples[i];

    solve(y_samples.data(), &result, a, b, n_samples);

    std::cout << std::fixed << std::setprecision(3) << result << std::endl;
}
