#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <hip/hip_runtime.h>
#include <float.h>
#include <cmath>

// Atomic max for float
__device__ void atomicMaxfloat(float *const addr, const float val) {
    if (*addr >= val) return;
    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

// 基本版本 - 直接原子操作
__global__ void findmax_basic(const float* input, float* globalmax, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    atomicMaxfloat(globalmax, input[idx]); 
}

__global__ void exponentialsum_basic(const float* input, float* output, int N, float globalmax, float* globalsum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    float val = expf(input[idx] - globalmax);
    output[idx] = val;
    atomicAdd(globalsum, val); 
}

// 優化版本 - 使用 shared memory
template<int BLOCK_SIZE>
__global__ void findmax_optimized(const float* input, float* globalmax, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    sdata[tidx] = (idx < N) ? input[idx] : -FLT_MAX;
    __syncthreads();
    
    for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] = fmax(sdata[tidx], sdata[tidx + i]);
        __syncthreads();
    }

    if (tidx == 0) atomicMaxfloat(globalmax, sdata[0]); 
}

template<int BLOCK_SIZE>
__global__ void exponentialsum_optimized(const float* input, float* output, int N, float globalmax, float* globalsum) {
    __shared__ float sdata[BLOCK_SIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    float val = (idx < N) ? expf(input[idx] - globalmax) : 0.0f;
    if (idx < N) output[idx] = val;

    sdata[tidx] = val;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }
    
    if (tidx == 0) atomicAdd(globalsum, sdata[0]); 
}

__global__ void softmax_normalize(float* output, int N, float globalsum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return; 
    output[idx] /= globalsum;
}

// 基本版本執行函數
double run_basic_version(const float* input, float* output, int N, int block_size) {
    float *d_input, *d_output, *d_globalmax, *d_globalsum;
    float globalmax, globalsum;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_globalmax, sizeof(float));
    hipMalloc(&d_globalsum, sizeof(float));

    float init_max = -FLT_MAX;
    float init_sum = 0.0f;
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalmax, &init_max, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalsum, &init_sum, sizeof(float), hipMemcpyHostToDevice);

    int blocksPerGrid = (N + block_size - 1) / block_size;

    auto start = std::chrono::high_resolution_clock::now();

    findmax_basic<<<blocksPerGrid, block_size>>>(d_input, d_globalmax, N);
    hipMemcpy(&globalmax, d_globalmax, sizeof(float), hipMemcpyDeviceToHost);

    hipMemcpy(d_globalsum, &init_sum, sizeof(float), hipMemcpyHostToDevice);
    exponentialsum_basic<<<blocksPerGrid, block_size>>>(d_input, d_output, N, globalmax, d_globalsum);
    hipMemcpy(&globalsum, d_globalsum, sizeof(float), hipMemcpyDeviceToHost);

    softmax_normalize<<<blocksPerGrid, block_size>>>(d_output, N, globalsum);
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0;

    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_globalmax);
    hipFree(d_globalsum);
    
    return exec_time;
}

// 優化版本執行函數
template<int BLOCK_SIZE>
double run_optimized_version(const float* input, float* output, int N) {
    float *d_input, *d_output, *d_globalmax, *d_globalsum;
    float globalmax, globalsum;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_globalmax, sizeof(float));
    hipMalloc(&d_globalsum, sizeof(float));

    float init_max = -FLT_MAX;
    float init_sum = 0.0f;
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalmax, &init_max, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_globalsum, &init_sum, sizeof(float), hipMemcpyHostToDevice);

    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    findmax_optimized<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(d_input, d_globalmax, N);
    hipMemcpy(&globalmax, d_globalmax, sizeof(float), hipMemcpyDeviceToHost);

    hipMemcpy(d_globalsum, &init_sum, sizeof(float), hipMemcpyHostToDevice);
    exponentialsum_optimized<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(d_input, d_output, N, globalmax, d_globalsum);
    hipMemcpy(&globalsum, d_globalsum, sizeof(float), hipMemcpyDeviceToHost);

    softmax_normalize<<<blocksPerGrid, BLOCK_SIZE>>>(d_output, N, globalsum);
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0;

    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_globalmax);
    hipFree(d_globalsum);
    
    return exec_time;
}

// CPU 版本用於驗證（不使用）
/*
void softmax_cpu(const float* input, float* output, int N) {
    // 找最大值
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // 計算 exp 和總和
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // 正規化
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

// 驗證結果正確性（不使用）
bool verify_results(const float* result1, const float* result2, int N, float tolerance = 1e-5f) {
    for (int i = 0; i < N; i++) {
        float diff = std::abs(result1[i] - result2[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}
*/

// 生成測試數據
void generate_test_data(std::vector<float>& data, int N) {
    std::random_device rd;
    std::mt19937 gen(42); // 固定種子
    std::normal_distribution<float> dis(0.0f, 2.0f); // 正態分布
    
    for (int i = 0; i < N; i++) {
        data[i] = dis(gen);
    }
}

void test_different_input_sizes() {
    std::cout << "=== 不同輸入大小效能測試 ===" << std::endl;
    
    std::vector<int> test_sizes = {10000000, 100000000, 1000000000};
    const int block_size = 256;
    
    std::cout << std::setw(12) << "輸入大小" 
              << std::setw(15) << "基本版本(ms)" 
              << std::setw(15) << "優化版本(ms)" 
              << std::setw(12) << "加速比" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (int N : test_sizes) {
        std::vector<float> input(N);
        std::vector<float> output_basic(N);
        std::vector<float> output_optimized(N);
        
        generate_test_data(input, N);
        
        // 執行測試
        double basic_time = run_basic_version(input.data(), output_basic.data(), N, block_size);
        double optimized_time = run_optimized_version<256>(input.data(), output_optimized.data(), N);
        
        double speedup = (optimized_time > 0) ? basic_time / optimized_time : 0.0;
        
        std::string size_str;
        if (N >= 1000000000) {
            size_str = std::to_string(N / 1000000000) + "B";
        } else if (N >= 1000000) {
            size_str = std::to_string(N / 1000000) + "M";
        } else if (N >= 1000) {
            size_str = std::to_string(N / 1000) + "K";
        } else {
            size_str = std::to_string(N);
        }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(12) << size_str
                  << std::setw(15) << basic_time
                  << std::setw(15) << optimized_time
                  << std::setw(12) << speedup << "x" << std::endl;
    }
    std::cout << std::endl;
}

void test_different_block_sizes() {
    std::cout << "=== 不同 Block Size 效能測試 ===" << std::endl;
    
    const int N = 100000000; // 100M elements
    std::vector<int> block_sizes = {128, 256, 512, 1024};
    
    std::cout << std::setw(12) << "Block Size" 
              << std::setw(15) << "基本版本(ms)" 
              << std::setw(15) << "優化版本(ms)" 
              << std::setw(12) << "加速比" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    std::vector<float> input(N);
    generate_test_data(input, N);
    
    for (int block_size : block_sizes) {
        std::vector<float> output_basic(N);
        std::vector<float> output_optimized(N);
        
        double basic_time = run_basic_version(input.data(), output_basic.data(), N, block_size);
        
        double optimized_time;
        switch(block_size) {
            case 128:
                optimized_time = run_optimized_version<128>(input.data(), output_optimized.data(), N);
                break;
            case 256:
                optimized_time = run_optimized_version<256>(input.data(), output_optimized.data(), N);
                break;
            case 512:
                optimized_time = run_optimized_version<512>(input.data(), output_optimized.data(), N);
                break;
            case 1024:
                optimized_time = run_optimized_version<1024>(input.data(), output_optimized.data(), N);
                break;
            default:
                continue;
        }
        
        double speedup = (optimized_time > 0) ? basic_time / optimized_time : 0.0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(12) << block_size
                  << std::setw(15) << basic_time
                  << std::setw(15) << optimized_time
                  << std::setw(12) << speedup << "x" << std::endl;
    }
    std::cout << std::endl;
}

void print_gpu_info() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    
    std::cout << "=== GPU 資訊 ===" << std::endl;
    std::cout << "裝置名稱: " << prop.name << std::endl;
    std::cout << "Compute Units: " << prop.multiProcessorCount << std::endl;
    std::cout << "最大 threads/block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory/block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "初始化 GPU..." << std::endl;
    
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "找不到 HIP 裝置！" << std::endl;
        return 1;
    }
    
    print_gpu_info();
    test_different_input_sizes();
    test_different_block_sizes();
    
    std::cout << "=== 分析結果 ===" << std::endl;
    std::cout << "基本版本問題：" << std::endl;
    std::cout << "- 大量原子操作競爭（2N 次）" << std::endl;
    std::cout << "- 記憶體存取效率低" << std::endl;
    std::cout << std::endl;
    std::cout << "優化版本優勢：" << std::endl;
    std::cout << "- 原子操作減少 256 倍（2N/256 次）" << std::endl;
    std::cout << "- 使用 shared memory 提升效率" << std::endl;
    std::cout << "- 更好的記憶體存取模式" << std::endl;
    
    return 0;
}