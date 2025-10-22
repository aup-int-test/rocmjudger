#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
using namespace std;

/* ============ Kernels ============ */

__global__ void matmul_kernel(const float *X, const float *beta, float *result,
                              int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float sum = 0.0f;
        for (int j = 0; j < n_features; j++)
            sum += X[idx * n_features + j] * beta[j];
        result[idx] = sum;
    }
}

__global__ void sigmoid_kernel(const float *z, float *p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        p[idx] = 1.0f / (1.0f + expf(-z[idx]));
}

__global__ void gradient_kernel(const float *X, const float *p, const float *y,
                                const float *beta, float *gradient,
                                int n_samples, int n_features, float lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float diff = p[i] - y[i];
            sum += X[i * n_features + idx] * diff;
        }
        gradient[idx] = (sum + lambda * beta[idx]) / (float)n_samples;
    }
}

__global__ void weights_kernel(const float *p, float *w, int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples)
        w[idx] = p[idx] * (1.0f - p[idx]);
}

__global__ void hessian_kernel(const float *X, const float *w, float *H,
                               int n_samples, int n_features, float reg) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_features && col < n_features) {
        float sum = 0.0f;
        for (int s = 0; s < n_samples; s++)
            sum += X[s * n_features + row] * w[s] * X[s * n_features + col];
        sum /= (float)n_samples;
        if (row == col) sum += reg;
        H[row * n_features + col] = sum;
    }
}

__global__ void newton_update_kernel(float *beta, const float *step, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_features)
        beta[idx] -= step[idx];
}

/* ============ CPU solve linear system ============ */

static void solve_linear_system(float *H, float *g, float *step, int n) {
    vector<float> aug(n * (n + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * (n + 1) + j] = H[i * n + j];
        aug[i * (n + 1) + n] = g[i];
    }

    for (int k = 0; k < n; k++) {
        int pivot = k;
        float maxv = fabsf(aug[k * (n + 1) + k]);
        for (int i = k + 1; i < n; i++) {
            float v = fabsf(aug[i * (n + 1) + k]);
            if (v > maxv) { maxv = v; pivot = i; }
        }
        if (pivot != k) {
            for (int j = 0; j <= n; j++)
                swap(aug[k * (n + 1) + j], aug[pivot * (n + 1) + j]);
        }
        float diag = aug[k * (n + 1) + k];
        if (fabsf(diag) < 1e-30f) diag = (diag >= 0.0f ? 1e-30f : -1e-30f);
        for (int i = k + 1; i < n; i++) {
            float f = aug[i * (n + 1) + k] / diag;
            for (int j = k; j <= n; j++)
                aug[i * (n + 1) + j] -= f * aug[k * (n + 1) + j];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        float acc = aug[i * (n + 1) + n];
        for (int j = i + 1; j < n; j++)
            acc -= aug[i * (n + 1) + j] * step[j];
        float den = aug[i * (n + 1) + i];
        if (fabsf(den) < 1e-30f) den = (den >= 0.0f ? 1e-30f : -1e-30f);
        step[i] = acc / den;
    }
}

/* ============ Training solve() ============ */

static void solve_logreg(const float *d_X, const float *d_y, float *d_beta,
                         int n_samples, int n_features) {
    const int max_iter = 30;
    const float lambda = 1e-6f;
    const float reg = 1e-6f;
    const int block = 256;

    hipMemset(d_beta, 0, (size_t)n_features * sizeof(float));

    float *d_z, *d_p, *d_w, *d_g, *d_H, *d_step;
    hipMalloc(&d_z, n_samples * sizeof(float));
    hipMalloc(&d_p, n_samples * sizeof(float));
    hipMalloc(&d_w, n_samples * sizeof(float));
    hipMalloc(&d_g, n_features * sizeof(float));
    hipMalloc(&d_H, (size_t)n_features * n_features * sizeof(float));
    hipMalloc(&d_step, n_features * sizeof(float));

    int grid_samples = (n_samples + block - 1) / block;
    int grid_features = (n_features + block - 1) / block;

    vector<float> h_H(n_features * n_features);
    vector<float> h_g(n_features);
    vector<float> h_step(n_features);

    dim3 bh(16, 16);
    dim3 gh((n_features + 15) / 16, (n_features + 15) / 16);

    for (int it = 0; it < max_iter; it++) {
        hipLaunchKernelGGL(matmul_kernel, dim3(grid_samples), dim3(block), 0, 0,
                           d_X, d_beta, d_z, n_samples, n_features);
        hipLaunchKernelGGL(sigmoid_kernel, dim3(grid_samples), dim3(block), 0, 0,
                           d_z, d_p, n_samples);
        hipLaunchKernelGGL(gradient_kernel, dim3(grid_features), dim3(block), 0, 0,
                           d_X, d_p, d_y, d_beta, d_g, n_samples, n_features, lambda);
        hipLaunchKernelGGL(weights_kernel, dim3(grid_samples), dim3(block), 0, 0,
                           d_p, d_w, n_samples);
        hipLaunchKernelGGL(hessian_kernel, gh, bh, 0, 0,
                           d_X, d_w, d_H, n_samples, n_features, reg);

        hipMemcpy(h_H.data(), d_H, (size_t)n_features * n_features * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(h_g.data(), d_g, n_features * sizeof(float), hipMemcpyDeviceToHost);

        solve_linear_system(h_H.data(), h_g.data(), h_step.data(), n_features);

        hipMemcpy(d_step, h_step.data(), n_features * sizeof(float), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(newton_update_kernel, dim3(grid_features), dim3(block), 0, 0,
                           d_beta, d_step, n_features);
    }

    hipFree(d_z);
    hipFree(d_p);
    hipFree(d_w);
    hipFree(d_g);
    hipFree(d_H);
    hipFree(d_step);
}

/* ============ main ============ */

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n_samples, n_features;
    cin >> n_samples >> n_features;
    if (n_samples < 1 || n_features < 1 || n_samples < n_features)
        return 0;

    vector<float> h_X((size_t)n_samples * n_features);
    vector<float> h_y(n_samples);
    for (int i = 0; i < n_samples * n_features; i++) cin >> h_X[i];
    for (int i = 0; i < n_samples; i++) {
        float v; cin >> v;
        h_y[i] = (v >= 0.5f) ? 1.0f : 0.0f;
    }

    float *d_X, *d_y, *d_beta;
    hipMalloc(&d_X, (size_t)n_samples * n_features * sizeof(float));
    hipMalloc(&d_y, n_samples * sizeof(float));
    hipMalloc(&d_beta, n_features * sizeof(float));

    hipMemcpy(d_X, h_X.data(), (size_t)n_samples * n_features * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y.data(), n_samples * sizeof(float), hipMemcpyHostToDevice);

    solve_logreg(d_X, d_y, d_beta, n_samples, n_features);

    vector<float> h_beta(n_features);
    hipMemcpy(h_beta.data(), d_beta, n_features * sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < n_features; i++) {
        if (i) cout << " ";
        cout.setf(ios::fixed);
        cout.precision(6);
        cout << h_beta[i];
    }
    cout << "\n";

    hipFree(d_X);
    hipFree(d_y);
    hipFree(d_beta);
    return 0;
}
