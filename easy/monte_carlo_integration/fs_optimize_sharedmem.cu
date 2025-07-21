#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

#define threadperblock 256

/* basic testcase
0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
*/

__global__ void montecarlo(const double* y_samples, double* result, double a, double b, int n_samples) {
    __shared__ double sdata[threadperblock];

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

    if (tidx == 0) atomicAdd(result, sdata[0]); 
}

void solve(const double* y_samples, double* result, double a, double b, int n_samples) {
    double* d_ysamples;
    double* d_result;

    hipMalloc(&d_ysamples, n_samples * sizeof(double));
    hipMalloc(&d_result, sizeof(double));

    hipMemcpy(d_ysamples, y_samples, n_samples * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_result, 0, sizeof(double));

    int blocksPerGrid = (n_samples + threadperblock - 1) / threadperblock;

    //size_t shared_mem_size = threadperblock * sizeof(double);
    montecarlo<<<blocksPerGrid, threadperblock>>>(d_ysamples, d_result, a, b, n_samples);
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

