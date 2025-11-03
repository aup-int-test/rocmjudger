#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

#define TILEX 32
#define TILEY 32

__global__ void batch_norm(const float* __restrict__ input,
                           const float* __restrict__ gamma,
                           const float* __restrict__ beta,
                           float* __restrict__ output,
                           int N, int C, float eps) {
    __shared__ float Ms[TILEY][TILEX]; // mean sum
    __shared__ float Ss[TILEY][TILEX]; // square sum

    int sample = threadIdx.y;
    int channel = threadIdx.x + blockDim.x * blockIdx.x;

    float m_sum = 0.0f;
    float s_sum = 0.0f;

    // step 1: accumulate mean and square sum
    for (int i = sample; i < N; i += blockDim.y) {
        if (channel < C) {
            float v = input[i * C + channel];
            m_sum += v;
            s_sum += v * v;
        }
    }

    Ms[threadIdx.y][threadIdx.x] = m_sum;
    Ss[threadIdx.y][threadIdx.x] = s_sum;
    __syncthreads();

    // step 2: reduction across Y
    for (int j = blockDim.y / 2; j > 0; j >>= 1) {
        if (threadIdx.y < j) {
            Ms[threadIdx.y][threadIdx.x] += Ms[threadIdx.y + j][threadIdx.x];
            Ss[threadIdx.y][threadIdx.x] += Ss[threadIdx.y + j][threadIdx.x];
        }
        __syncthreads();
    }

    // step 3: mean, variance, inv_std
    float mean = Ms[0][threadIdx.x] / static_cast<float>(N);
    float var = Ss[0][threadIdx.x] / static_cast<float>(N) - mean * mean;
    float inv_std = rsqrtf(var + eps);

    float g = (channel < C) ? gamma[channel] : 0.0f;
    float b = (channel < C) ? beta[channel] : 0.0f;

    // step 4: normalize
    for (int i = sample; i < N; i += blockDim.y) {
        if (channel < C) {
            float x = input[i * C + channel];
            output[i * C + channel] = g * ((x - mean) * inv_std) + b;
        }
    }
}

void solve(const float* input, const float* gamma, const float* beta,
           float* output, int N, int C, float eps) {
    dim3 threadsPerBlock(TILEX, TILEY);
    dim3 blocksPerGrid((C + TILEX - 1) / TILEX, 1);
    hipLaunchKernelGGL(batch_norm, blocksPerGrid, threadsPerBlock, 0, 0,
                       input, gamma, beta, output, N, C, eps);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << (argc ? argv[0] : "prog")
                  << " <input_file>\n";
        return 1;
    }
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr << "Error: cannot open " << argv[1] << "\n";
        return 1;
    }

    int N, C;
    float eps;
    fin >> N >> C >> eps;

    std::vector<float> input(N * C);
    std::vector<float> gamma(C);
    std::vector<float> beta(C);
    std::vector<float> output(N * C);

    for (int i = 0; i < N * C; ++i)
        fin >> input[i];
    for (int i = 0; i < C; ++i)
        fin >> gamma[i];
    for (int i = 0; i < C; ++i)
        fin >> beta[i];
    fin.close();

    float *d_input, *d_gamma, *d_beta, *d_output;
    hipMalloc(&d_input, N * C * sizeof(float));
    hipMalloc(&d_gamma, C * sizeof(float));
    hipMalloc(&d_beta, C * sizeof(float));
    hipMalloc(&d_output, N * C * sizeof(float));

    hipMemcpy(d_input, input.data(), N * C * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gamma, gamma.data(), C * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_beta, beta.data(), C * sizeof(float), hipMemcpyHostToDevice);

    solve(d_input, d_gamma, d_beta, d_output, N, C, eps);
    hipDeviceSynchronize();

    hipMemcpy(output.data(), d_output, N * C * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            if (j) std::cout << " ";
            std::cout << output[i * C + j];
        }
        std::cout << "\n";
    }

    hipFree(d_input);
    hipFree(d_gamma);
    hipFree(d_beta);
    hipFree(d_output);
    return 0;
}
