#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <hip/hip_runtime.h>

// 基本版本 - 沒有優化
__global__ void convolution2D_basic(const int* input, const int* kernel, int* output,
                                   int input_rows, int input_cols, 
                                   int kernel_rows, int kernel_cols,
                                   int output_rows, int output_cols) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= output_rows || col >= output_cols) return;
    
    int sum = 0;
    
    for (int kernel_r = 0; kernel_r < kernel_rows; kernel_r++) {
        for (int kernel_c = 0; kernel_c < kernel_cols; kernel_c++) {
            int input_r = row + kernel_r;
            int input_c = col + kernel_c;
            
            if (input_r < input_rows && input_c < input_cols) {
                sum += input[input_r * input_cols + input_c] * kernel[kernel_r * kernel_cols + kernel_c];
            }
        }
    }
    
    output[row * output_cols + col] = sum;
}

// 優化版本 - 使用 shared memory (halo-based tiling)
__constant__ int c_kernel[256*256]; // 支援最大 32x32 kernel

__global__ void convolution2D_optimized(const int* input, int* output,
                                       int input_rows, int input_cols, 
                                       int kernel_rows, int kernel_cols,
                                       int output_rows, int output_cols) {
    
    // 計算 tile 大小（包含 halo）
    int tile_rows = blockDim.y + kernel_rows - 1;
    int tile_cols = blockDim.x + kernel_cols - 1;
    
    __shared__ int s_input[8132];
    
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tile_start_row = blockIdx.y * blockDim.y;
    int tile_start_col = blockIdx.x * blockDim.x;
    
    // 載入 input tile（包含 halo）到 shared memory
    for (int i = local_row; i < tile_rows; i += blockDim.y) {
        for (int j = local_col; j < tile_cols; j += blockDim.x) {
            int input_row = tile_start_row + i;
            int input_col = tile_start_col + j;
            
            if (input_row < input_rows && input_col < input_cols) {
                s_input[i * tile_cols + j] = input[input_row * input_cols + input_col];
            } else {
                s_input[i * tile_cols + j] = 0;  // padding
            }
        }
    }
    
    __syncthreads();
    
    // 計算卷積
    if (global_row < output_rows && global_col < output_cols) {
        int sum = 0;
        
        for (int kr = 0; kr < kernel_rows; kr++) {
            for (int kc = 0; kc < kernel_cols; kc++) {
                int shared_row = local_row + kr;
                int shared_col = local_col + kc;
                sum += s_input[shared_row * tile_cols + shared_col] * 
                       c_kernel[kr * kernel_cols + kc];
            }
        }
        
        output[global_row * output_cols + global_col] = sum;
    }
}

// 基本版本的執行函數
double run_basic_version(const int* input, const int* kernel, int* output,
                        int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t input_size = input_rows * input_cols * sizeof(int);
    size_t kernel_size = kernel_rows * kernel_cols * sizeof(int);
    size_t output_size = output_rows * output_cols * sizeof(int);

    int *d_input, *d_kernel, *d_output;

    hipMalloc(&d_input, input_size);
    hipMalloc(&d_kernel, kernel_size);
    hipMalloc(&d_output, output_size);

    hipMemcpy(d_input, input, input_size, hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel, kernel_size, hipMemcpyHostToDevice);

    dim3 threads(16, 16);  
    dim3 blocks((output_cols + threads.x - 1) / threads.x, 
                (output_rows + threads.y - 1) / threads.y);

    // 計時開始
    auto start = std::chrono::high_resolution_clock::now();
    
    convolution2D_basic<<<blocks, threads>>>(d_input, d_kernel, d_output,
                                            input_rows, input_cols, kernel_rows, kernel_cols,
                                            output_rows, output_cols);
    
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0; // 轉換為毫秒
    
    hipMemcpy(output, d_output, output_size, hipMemcpyDeviceToHost);
    
    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);
    
    return exec_time;
}

// 優化版本的執行函數
double run_optimized_version(const int* input, const int* kernel, int* output,
                           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t input_size = input_rows * input_cols * sizeof(int);
    size_t kernel_size = kernel_rows * kernel_cols * sizeof(int);
    size_t output_size = output_rows * output_cols * sizeof(int);


    int *d_input, *d_output;

    hipMalloc(&d_input, input_size);
    hipMalloc(&d_output, output_size);

    hipMemcpy(d_input, input, input_size, hipMemcpyHostToDevice);
    
    // 複製 kernel 到 constant memory
    hipMemcpyToSymbol(c_kernel, kernel, kernel_size);

    dim3 threads(16, 16);
    dim3 blocks((output_cols + threads.x - 1) / threads.x, 
                (output_rows + threads.y - 1) / threads.y);

    // 計算 shared memory 大小
    int tile_rows = threads.y + kernel_rows - 1;
    int tile_cols = threads.x + kernel_cols - 1;
    int shared_size = tile_rows * tile_cols * sizeof(int);
    

    // 計時開始
    auto start = std::chrono::high_resolution_clock::now();
    
    convolution2D_optimized<<<blocks, threads>>>(d_input, d_output,
                                                             input_rows, input_cols, 
                                                             kernel_rows, kernel_cols,
                                                             output_rows, output_cols);
    
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0; // 轉換為毫秒
    
    hipMemcpy(output, d_output, output_size, hipMemcpyDeviceToHost);
    
    hipFree(d_input);
    hipFree(d_output);
    
    return exec_time;
}

// 驗證結果正確性
bool verify_results(const std::vector<int>& result1, const std::vector<int>& result2) {
    if (result1.size() != result2.size()) return false;
    
    for (size_t i = 0; i < result1.size(); i++) {
        if (result1[i] != result2[i]) {
            return false;
        }
    }
    return true;
}

// CPU 版本用於驗證
void convolution2D_cpu(const int* input, const int* kernel, int* output,
                      int input_rows, int input_cols, 
                      int kernel_rows, int kernel_cols) {
    
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    for (int out_r = 0; out_r < output_rows; out_r++) {
        for (int out_c = 0; out_c < output_cols; out_c++) {
            int sum = 0;
            for (int kr = 0; kr < kernel_rows; kr++) {
                for (int kc = 0; kc < kernel_cols; kc++) {
                    sum += input[(out_r + kr) * input_cols + (out_c + kc)] * 
                           kernel[kr * kernel_cols + kc];
                }
            }
            output[out_r * output_cols + out_c] = sum;
        }
    }
}

// 效能測試套件
void benchmark_comparison() {
    std::cout << "=== 卷積效能比較測試 ===" << std::endl;
    
    // 測試案例
    std::vector<std::tuple<int, int, int, int, std::string>> test_cases = {

        {512, 512, 256, 256, "小 kernel, 中等輸入"},
        {1024, 1024, 256, 256, "小 kernel, 大輸入"},
        {2048, 2048, 256, 256, "小 kernel, 超大輸入"},
    };
    
    std::cout << std::setw(25) << "測試案例" 
              << std::setw(15) << "基本版本(ms)" 
              << std::setw(15) << "優化版本(ms)" 
              << std::setw(12) << "加速比" 
              << std::setw(8) << "正確性" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (auto& [input_rows, input_cols, kernel_rows, kernel_cols, description] : test_cases) {
        int output_rows = input_rows - kernel_rows + 1;
        int output_cols = input_cols - kernel_cols + 1;
        
        // 生成測試數據
        std::vector<int> input(input_rows * input_cols);
        std::vector<int> kernel(kernel_rows * kernel_cols);
        std::vector<int> basic_output(output_rows * output_cols);
        std::vector<int> optimized_output(output_rows * output_cols);
        std::vector<int> cpu_output(output_rows * output_cols);
        
        // 初始化隨機數據
        for (int i = 0; i < input_rows * input_cols; i++) {
            input[i] = rand() % 10 - 5;  // -5 到 4 的隨機數
        }
        for (int i = 0; i < kernel_rows * kernel_cols; i++) {
            kernel[i] = rand() % 3 - 1;  // -1 到 1 的隨機數
        }
        
        // 執行基本版本
        double basic_time = run_basic_version(input.data(), kernel.data(), basic_output.data(),
                                            input_rows, input_cols, kernel_rows, kernel_cols);
        
        // 執行優化版本
        double optimized_time = run_optimized_version(input.data(), kernel.data(), optimized_output.data(),
                                                    input_rows, input_cols, kernel_rows, kernel_cols);
        
        // CPU 版本驗證（只對小測試案例）
        bool correct = true;
        convolution2D_cpu(input.data(), kernel.data(), cpu_output.data(),
                            input_rows, input_cols, kernel_rows, kernel_cols);
        correct = verify_results(basic_output, cpu_output) && 
                    verify_results(optimized_output, cpu_output);

        
        // 計算加速比
        double speedup = (optimized_time > 0) ? basic_time / optimized_time : 0.0;
        
        // 輸出結果
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(25) << description
                  << std::setw(15) << basic_time
                  << std::setw(15) << optimized_time
                  << std::setw(12) << speedup << "x"
                  << std::setw(8) << (correct ? "✓" : "✗") << std::endl;
        
        // 如果結果不正確，顯示詳細資訊
        if (!correct) {
            std::cout << "  警告：結果驗證失敗！" << std::endl;
            if (output_rows * output_cols <= 25) {
                std::cout << "  基本版本前幾個結果: ";
                for (int i = 0; i < std::min(5, (int)basic_output.size()); i++) {
                    std::cout << basic_output[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "  優化版本前幾個結果: ";
                for (int i = 0; i < std::min(5, (int)optimized_output.size()); i++) {
                    std::cout << optimized_output[i] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
}

// GPU 硬體資訊
void print_gpu_info() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    
    std::cout << "=== GPU 硬體資訊 ===" << std::endl;
    std::cout << "裝置名稱: " << prop.name << std::endl;
    std::cout << "計算單元數: " << prop.multiProcessorCount << std::endl;
    std::cout << "最大線程/塊: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory/塊: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Constant Memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "全域記憶體: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "記憶體時鐘: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "記憶體匯流排寬度: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "初始化 GPU..." << std::endl;
    
    // 檢查 GPU 可用性
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "找不到 HIP 裝置！" << std::endl;
        return 1;
    }
    
    print_gpu_info();
    benchmark_comparison();
    
    std::cout << "=== 優化技術說明 ===" << std::endl;
    std::cout << "基本版本：" << std::endl;
    std::cout << "  - 每個 thread 直接從全域記憶體讀取 input 和 kernel" << std::endl;
    std::cout << "  - 沒有記憶體快取優化" << std::endl;
    std::cout << std::endl;
    std::cout << "優化版本：" << std::endl;
    std::cout << "  - Kernel 存放在 constant memory (快速且所有 threads 共享)" << std::endl;
    std::cout << "  - Input 使用 shared memory tiling (halo-based)" << std::endl;
    std::cout << "  - 減少全域記憶體存取次數" << std::endl;
    std::cout << "  - 利用記憶體階層提升效能" << std::endl;
    
    return 0;
}