#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

__global__ void max_pooling_2d(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N, int C, int H, int W,
                               int H_out, int W_out,
                               int kernel_size, int stride, int padding) {
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    if (n >= N) return;

    int h_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_out_idx >= H_out || w_out_idx >= W_out) return;

    int h_start = h_out_idx * stride - padding;
    int w_start = w_out_idx * stride - padding;
    int input_start = (n * C + c) * (H * W);

    float val = -INFINITY;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_idx = h_start + kh;
            int w_idx = w_start + kw;

            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                int input_idx = input_start + h_idx * W + w_idx;
                val = fmaxf(val, input[input_idx]);
            }
        }
    }

    output[(n * C + c) * (H_out * W_out) + h_out_idx * W_out + w_out_idx] = val;
}

void solve(const float* input, float* output,
           int N, int C, int H, int W,
           int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (W_out + blockDim.x - 1) / blockDim.x,
        (H_out + blockDim.y - 1) / blockDim.y,
        N * C
    );

    hipLaunchKernelGGL(max_pooling_2d, gridDim, blockDim, 0, 0,
                       input, output, N, C, H, W, H_out, W_out,
                       kernel_size, stride, padding);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N, C, H, W;
    int kernel_size, stride, padding;
    std::cin >> N >> C >> H >> W >> kernel_size >> stride >> padding;

    int input_size = N * C * H * W;
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    int output_size = N * C * H_out * W_out;

    std::vector<float> input(input_size);
    for (int i = 0; i < input_size; ++i)
        std::cin >> input[i];

    std::vector<float> output(output_size, 0.0f);

    float *d_input, *d_output;
    hipMalloc(&d_input, input_size * sizeof(float));
    hipMalloc(&d_output, output_size * sizeof(float));

    hipMemcpy(d_input, input.data(), input_size * sizeof(float), hipMemcpyHostToDevice);

    solve(d_input, d_output, N, C, H, W, kernel_size, stride, padding);

    hipMemcpy(output.data(), d_output, output_size * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    if (w) std::cout << " ";
                    std::cout << output[((n * C + c) * H_out + h) * W_out + w];
                }
                std::cout << "\n";
            }
            if (n != N - 1 || c != C - 1) std::cout << "\n";
        }
    }

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
