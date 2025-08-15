#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

__global__ void montecarlo(const double* y_samples, double* result, double a, double b, int n_samples){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n_samples) return;

    double tmp = (b - a) * y_samples[idx] / n_samples;

    atomicAdd(result, tmp); 
}

void solve(const double* y_samples, double* result, double a, double b, int n_samples) {
    double *d_ysamples, *d_result;

    hipMalloc(&d_ysamples, n_samples * sizeof(double));
    hipMalloc(&d_result, sizeof(double));

    hipMemcpy(d_ysamples, y_samples, n_samples * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_result, 0, sizeof(double));

    int threadperblock = 256;
    int blockpergrid = (n_samples + threadperblock - 1) / threadperblock;

    montecarlo<<<blockpergrid, threadperblock>>>(d_ysamples, d_result, a, b, n_samples);
    hipDeviceSynchronize();

    hipMemcpy(result, d_result, sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_ysamples);
    hipFree(d_result);
}

int main(){
    int n_samples;
    double a, b, result;
    std::cin >> a >> b >> n_samples;

    std::vector<double> y_samples(n_samples);

    for(int i = 0; i < n_samples; ++i) std::cin >> y_samples[i];

    solve(y_samples.data(), &result, a, b, n_samples);

    std::cout << result << std::endl;
}
