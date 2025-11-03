// max_pooling_2d_arg_hip.hip
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cfloat>  // FLT_MAX

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
    int input_base = (n * C + c) * (H * W);

    float best = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_idx = h_start + kh;
            int w_idx = w_start + kw;
            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                int idx = input_base + h_idx * W + w_idx;
                best = fmaxf(best, input[idx]);
            }
        }
    }

    output[(n * C + c) * (H_out * W_out) + h_out_idx * W_out + w_out_idx] = best;
}

void solve(const float* input, float* output,
           int N, int C, int H, int W,
           int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C
    );

    hipLaunchKernelGGL(max_pooling_2d, grid, block, 0, 0,
                       input, output, N, C, H, W, H_out, W_out,
                       kernel_size, stride, padding);
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

    int N, C, H, W, kernel_size, stride, padding;
    fin >> N >> C >> H >> W >> kernel_size >> stride >> padding;

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    const int in_elems  = N * C * H * W;
    const int out_elems = N * C * H_out * W_out;

    std::vector<float> h_in(in_elems);
    for (int i = 0; i < in_elems; ++i) fin >> h_in[i];
    fin.close();

    float *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in,  in_elems  * sizeof(float));
    hipMalloc(&d_out, out_elems * sizeof(float));
    hipMemcpy(d_in, h_in.data(), in_elems * sizeof(float), hipMemcpyHostToDevice);

    solve(d_in, d_out, N, C, H, W, kernel_size, stride, padding);
    hipDeviceSynchronize();

    std::vector<float> h_out(out_elems);
    hipMemcpy(h_out.data(), d_out, out_elems * sizeof(float), hipMemcpyDeviceToHost);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    if (w) std::cout << " ";
                    std::cout << h_out[((n * C + c) * H_out + h) * W_out + w];
                }
                std::cout << "\n";
            }
            if (n != N - 1 || c != C - 1) std::cout << "\n";
        }
    }

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
