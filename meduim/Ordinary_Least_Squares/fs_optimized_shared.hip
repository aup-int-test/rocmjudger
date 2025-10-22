#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

__global__ void compute_xtx(const float* __restrict__ X,
                            float* __restrict__ XtX,
                            int n_samples, int n_features) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_features || j >= n_features) return;
    float sum = 0.0f;
    for (int k = 0; k < n_samples; ++k)
        sum += X[k * n_features + i] * X[k * n_features + j];
    XtX[i * n_features + j] = sum;
}

__global__ void compute_xty(const float* __restrict__ X,
                            const float* __restrict__ y,
                            float* __restrict__ Xty,
                            int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_features) return;
    float sum = 0.0f;
    for (int k = 0; k < n_samples; ++k)
        sum += X[k * n_features + i] * y[k];
    Xty[i] = sum;
}

void cpu_gauss_jordan(vector<float>& A, vector<float>& b, int n) {
    for (int i = 0; i < n; ++i) {
        float pivot = A[i * n + i];
        if (fabs(pivot) < 1e-8f) pivot = (pivot >= 0 ? 1e-8f : -1e-8f);
        for (int j = 0; j < n; ++j) A[i * n + j] /= pivot;
        b[i] /= pivot;
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            float factor = A[k * n + i];
            for (int j = 0; j < n; ++j)
                A[k * n + j] -= factor * A[i * n + j];
            b[k] -= factor * b[i];
        }
    }
}

extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {
    float *d_XtX, *d_Xty;
    hipMalloc(&d_XtX, sizeof(float) * n_features * n_features);
    hipMalloc(&d_Xty, sizeof(float) * n_features);
    dim3 block(16, 16);
    dim3 grid((n_features + 15) / 16, (n_features + 15) / 16);
    hipLaunchKernelGGL(compute_xtx, grid, block, 0, 0, X, d_XtX, n_samples, n_features);
    int threads = 256;
    int blocks = (n_features + threads - 1) / threads;
    hipLaunchKernelGGL(compute_xty, dim3(blocks), dim3(threads), 0, 0, X, y, d_Xty, n_samples, n_features);
    vector<float> h_XtX(n_features * n_features);
    vector<float> h_Xty(n_features);
    hipMemcpy(h_XtX.data(), d_XtX, sizeof(float) * n_features * n_features, hipMemcpyDeviceToHost);
    hipMemcpy(h_Xty.data(), d_Xty, sizeof(float) * n_features, hipMemcpyDeviceToHost);
    cpu_gauss_jordan(h_XtX, h_Xty, n_features);
    hipMemcpy(beta, h_Xty.data(), sizeof(float) * n_features, hipMemcpyHostToDevice);
    hipFree(d_XtX);
    hipFree(d_Xty);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    ifstream fin(argv[1]);
    int n_samples, n_features;
    fin >> n_samples >> n_features;
    vector<float> hX(n_samples * n_features);
    vector<float> hy(n_samples);
    for (int i = 0; i < n_samples * n_features; ++i) fin >> hX[i];
    for (int i = 0; i < n_samples; ++i) fin >> hy[i];
    float *dX, *dY, *dBeta;
    hipMalloc(&dX, sizeof(float) * hX.size());
    hipMalloc(&dY, sizeof(float) * hy.size());
    hipMalloc(&dBeta, sizeof(float) * n_features);
    hipMemcpy(dX, hX.data(), sizeof(float) * hX.size(), hipMemcpyHostToDevice);
    hipMemcpy(dY, hy.data(), sizeof(float) * hy.size(), hipMemcpyHostToDevice);
    solve(dX, dY, dBeta, n_samples, n_features);
    vector<float> hBeta(n_features);
    hipMemcpy(hBeta.data(), dBeta, sizeof(float) * n_features, hipMemcpyDeviceToHost);
    for (int i = 0; i < n_features; ++i)
        cout << hBeta[i] << (i + 1 == n_features ? '\n' : ' ');
    hipFree(dX);
    hipFree(dY);
    hipFree(dBeta);
    return 0;
}
