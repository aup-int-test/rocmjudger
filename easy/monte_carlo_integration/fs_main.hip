#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    std::ifstream input_file;
    std::string filename = argv[1];
    
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }
    int n_samples;
    double a, b, result;
    input_file >> a >> b >> n_samples;

    std::vector<double> y_samples(n_samples);

    for(int i = 0; i < n_samples; ++i) input_file >> y_samples[i];

    input_file.close();

    solve(y_samples.data(), &result, a, b, n_samples);

    std::cout << std::fixed << result << std::endl;
}
