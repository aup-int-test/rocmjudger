#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <hip/hip_runtime.h>
#include <float.h>
#include <cmath>

#define TILE_SIZE 16

// 基本版本 kernels
__global__ void computescore_basic(const float *Q, const float *K, float *QKT, int M, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += Q[row * d + i] * K[col * d + i];
        }
        QKT[row * N + col] = sum / sqrtf((float)d);
    }
}

__global__ void computeresult_basic(const float *QKT, const float *V, float *output, int M, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += QKT[row * N + k] * V[k * d + col];
        }
        output[row * d + col] = sum;
    }
}

// 優化版本 kernels
__global__ void computescore_optimized(const float *Q, const float *K, float *QKT, int M, int N, int d) {
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (d + TILE_SIZE - 1) / TILE_SIZE; k++) {
        int q_col = k * TILE_SIZE + tx;
        if (row < M && q_col < d) {
            tile_Q[ty][tx] = Q[row * d + q_col];
        } else {
            tile_Q[ty][tx] = 0.0f;
        }
        
        int k_col = k * TILE_SIZE + ty;
        if (col < N && k_col < d) {
            tile_K[tx][ty] = K[col * d + k_col];
        } else {
            tile_K[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_Q[ty][i] * tile_K[tx][i];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        QKT[row * N + col] = sum / sqrtf((float)d);
    }
}

__global__ void computeresult_optimized(const float *QKT, const float *V, float *output, int M, int N, int d) {
    __shared__ float tile_QKT[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; k++) {
        int qkt_col = k * TILE_SIZE + tx;
        if (row < M && qkt_col < N) {
            tile_QKT[ty][tx] = QKT[row * N + qkt_col];
        } else {
            tile_QKT[ty][tx] = 0.0f;
        }
        
        int v_row = k * TILE_SIZE + ty;
        if (v_row < N && col < d) {
            tile_V[ty][tx] = V[v_row * d + col];
        } else {
            tile_V[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_QKT[ty][i] * tile_V[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < d) {
        output[row * d + col] = sum;
    }
}

// 共用的 softmax kernel
__global__ void softmax(float *QKT, int M, int N) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (row < M) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, QKT[row * N + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            QKT[row * N + j] = expf(QKT[row * N + j] - max_val);
            sum_exp += QKT[row * N + j];
        }
        
        for (int j = 0; j < N; ++j) {
            QKT[row * N + j] /= sum_exp;
        }
    }
}

// 基本版本執行函數
double run_basic_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *d_Q, *d_K, *d_V, *d_QKT, *d_output;

    hipMalloc(&d_Q, M * d * sizeof(float));
    hipMalloc(&d_K, N * d * sizeof(float));
    hipMalloc(&d_V, N * d * sizeof(float));
    hipMalloc(&d_QKT, M * N * sizeof(float));
    hipMalloc(&d_output, M * d * sizeof(float));

    hipMemcpy(d_Q, Q, M * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, N * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, N * d * sizeof(float), hipMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // 固定使用 16×16 tiles
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    computescore_basic<<<blocks, threads>>>(d_Q, d_K, d_QKT, M, N, d);
    
    // 固定使用 256 threads 的 softmax
    softmax<<<M, 256>>>(d_QKT, M, N);

    blocks = dim3((d + 15) / 16, (M + 15) / 16);
    computeresult_basic<<<blocks, threads>>>(d_QKT, d_V, d_output, M, N, d);
    
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0;

    hipMemcpy(output, d_output, M * d * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_QKT);
    hipFree(d_output);
    
    return exec_time;
}

// 優化版本執行函數
double run_optimized_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *d_Q, *d_K, *d_V, *d_QKT, *d_output;

    hipMalloc(&d_Q, M * d * sizeof(float));
    hipMalloc(&d_K, N * d * sizeof(float));
    hipMalloc(&d_V, N * d * sizeof(float));
    hipMalloc(&d_QKT, M * N * sizeof(float));
    hipMalloc(&d_output, M * d * sizeof(float));

    hipMemcpy(d_Q, Q, M * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, N * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, N * d * sizeof(float), hipMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // 固定使用 16×16 tiles
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    computescore_optimized<<<blocks, threads>>>(d_Q, d_K, d_QKT, M, N, d);
    
    // 固定使用 256 threads 的 softmax
    softmax<<<M, 256>>>(d_QKT, M, N);

    blocks = dim3((d + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    computeresult_optimized<<<blocks, threads>>>(d_QKT, d_V, d_output, M, N, d);
    
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0;

    hipMemcpy(output, d_output, M * d * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_QKT);
    hipFree(d_output);
    
    return exec_time;
}

void generate_test_data(std::vector<float>& data, int size) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

bool verify_results(const float* result1, const float* result2, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(result1[i] - result2[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== Attention Shared Memory 優化測試 ===" << std::endl;
    std::cout << "固定配置: Tile 16×16, Softmax 256 threads" << std::endl;
    std::cout << std::endl;
    
    // 測試不同規模
    std::vector<std::tuple<int, int, int, std::string>> test_cases = {
        {512, 512, 64, "512×512, d=64"},
        {1024, 1024, 128, "1024×1024, d=128"},
        {2048, 2048, 256, "2048×2048, d=256"},
        {4096, 4096, 512, "4096×4096, d=512"}
    };
    
    std::cout << std::setw(20) << "測試規模" 
              << std::setw(15) << "基本版本(ms)" 
              << std::setw(15) << "優化版本(ms)" 
              << std::setw(12) << "加速比" 
              << std::setw(8) << "正確性" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (auto& [M, N, d, desc] : test_cases) {
        // 生成測試數據
        std::vector<float> Q(M * d), K(N * d), V(N * d);
        std::vector<float> output_basic(M * d), output_optimized(M * d);
        
        generate_test_data(Q, M * d);
        generate_test_data(K, N * d);
        generate_test_data(V, N * d);
        
        // 執行基本版本
        double basic_time = run_basic_attention(Q.data(), K.data(), V.data(), output_basic.data(), M, N, d);
        
        // 執行優化版本
        double optimized_time = run_optimized_attention(Q.data(), K.data(), V.data(), output_optimized.data(), M, N, d);
        
        // 驗證正確性
        bool correct = verify_results(output_basic.data(), output_optimized.data(), M * d);
        
        // 計算加速比
        double speedup = (optimized_time > 0) ? basic_time / optimized_time : 0.0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(20) << desc
                  << std::setw(15) << basic_time
                  << std::setw(15) << optimized_time
                  << std::setw(12) << speedup << "x"
                  << std::setw(8) << (correct ? "✓" : "✗") << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "優化技術: 使用 shared memory tiling 減少 global memory 存取" << std::endl;
    
    return 0;
}