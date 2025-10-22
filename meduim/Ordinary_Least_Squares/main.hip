#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

static __global__ void compute_xtx(const float* __restrict__ X,
                                   float* __restrict__ XtX,
                                   int n_samples, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_features || j >= n_features) return;
    float sum = 0.0f;
    for (int k = 0; k < n_samples; ++k) {
        float xi = X[k * n_features + i];
        float xj = X[k * n_features + j];
        sum += xi * xj;
    }
    XtX[i * n_features + j] = sum;
}

static __global__ void compute_xty(const float* __restrict__ X,
                                   const float* __restrict__ y,
                                   float* __restrict__ Xty,
                                   int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_features) return;
    float sum = 0.0f;
    for (int k = 0; k < n_samples; ++k) sum += X[k * n_features + i] * y[k];
    Xty[i] = sum;
}

static __global__ void gauss_jordan_solve(float* __restrict__ A,
                                          float* __restrict__ b,
                                          int n) {
    int j = threadIdx.x;
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        if (fabsf(pivot) < 1e-20f) pivot = (pivot >= 0.0f ? 1e-20f : -1e-20f);
        if (j < n) A[k * n + j] /= pivot;
        if (j == 0) b[k] /= pivot;
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            float factor = A[i * n + k];
            if (j < n) A[i * n + j] -= factor * A[k * n + j];
            if (j == 0) b[i] -= factor * b[k];
            __syncthreads();
        }
    }
}

extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {
    float* d_XtX = nullptr;
    float* d_Xty = nullptr;
    hipMalloc(&d_XtX, sizeof(float) * n_features * n_features);
    hipMalloc(&d_Xty, sizeof(float) * n_features);
    dim3 block2d(16, 16);
    dim3 grid2d((n_features + block2d.x - 1) / block2d.x,
                (n_features + block2d.y - 1) / block2d.y);
    hipLaunchKernelGGL(compute_xtx, grid2d, block2d, 0, 0, X, d_XtX, n_samples, n_features);
    int bsz = 256;
    int gsz = (n_features + bsz - 1) / bsz;
    hipLaunchKernelGGL(compute_xty, dim3(gsz), dim3(bsz), 0, 0, X, y, d_Xty, n_samples, n_features);
    int threads = n_features;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;
    hipLaunchKernelGGL(gauss_jordan_solve, dim3(1), dim3(threads), 0, 0, d_XtX, d_Xty, n_features);
    hipMemcpy(beta, d_Xty, sizeof(float) * n_features, hipMemcpyDeviceToDevice);
    hipFree(d_XtX);
    hipFree(d_Xty);
}

int main() {
    int n_samples, n_features;
    cin >> n_samples >> n_features;
    vector<float> hX(n_samples * n_features);
    vector<float> hy(n_samples);
    for (int i = 0; i < n_samples * n_features; ++i) cin >> hX[i];
    for (int i = 0; i < n_samples; ++i) cin >> hy[i];
    float *dX, *dY, *dBeta;
    hipMalloc(&dX, sizeof(float) * hX.size());
    hipMalloc(&dY, sizeof(float) * hy.size());
    hipMalloc(&dBeta, sizeof(float) * n_features);
    hipMemcpy(dX, hX.data(), sizeof(float) * hX.size(), hipMemcpyHostToDevice);
    hipMemcpy(dY, hy.data(), sizeof(float) * hy.size(), hipMemcpyHostToDevice);
    solve(dX, dY, dBeta, n_samples, n_features);
    vector<float> hBeta(n_features);
    hipMemcpy(hBeta.data(), dBeta, sizeof(float) * n_features, hipMemcpyDeviceToHost);
    for (int i = 0; i < n_features; ++i) cout << hBeta[i] << (i + 1 == n_features ? '\n' : ' ');
    hipFree(dX);
    hipFree(dY);
    hipFree(dBeta);
    return 0;
}
