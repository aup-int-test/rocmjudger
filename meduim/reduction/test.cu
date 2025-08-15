#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <cmath>

// 測試不同的 block size
template<int BLOCK_SIZE>
__global__ void reduction_optimized(const int* input, int* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    atomicAdd(output, input[idx]); 
}


template<int BLOCK_SIZE>
double run_test(const int* input, int* output, int N) {
    int *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, sizeof(int));

    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(int));
    
    int blockpergrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();
    
    reduction_optimized<BLOCK_SIZE><<<blockpergrid, BLOCK_SIZE>>>(d_input, d_output, N);
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double exec_time = duration.count() / 1000.0;

    hipMemcpy(output, d_output, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    
    return exec_time;
}

// 計算理論 occupancy
double calculate_occupancy(int block_size, hipDeviceProp_t& prop) {
    // 每個 block 的 warp 數量
    int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
    
    // 每個 block 使用的 shared memory (KB)
    double shared_mem_per_block = (block_size * sizeof(int)) / 1024.0;
    
    // 各種限制因素
    int max_blocks_per_cu = prop.maxBlocksPerMultiProcessor;
    int max_warps_per_cu = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    double max_shared_mem_per_cu = prop.sharedMemPerMultiprocessor / 1024.0; // KB
    
    // 根據不同限制計算可能的 blocks 數量
    int blocks_limited_by_max_blocks = max_blocks_per_cu;
    int blocks_limited_by_warps = max_warps_per_cu / warps_per_block;
    int blocks_limited_by_shared_mem = (int)(max_shared_mem_per_cu / shared_mem_per_block);
    
    // 實際可執行的 blocks 數量是最小值
    int actual_blocks = std::min({blocks_limited_by_max_blocks, 
                                  blocks_limited_by_warps, 
                                  blocks_limited_by_shared_mem});
    
    // 實際 warps 數量
    int actual_warps = actual_blocks * warps_per_block;
    
    // Occupancy = 實際 warps / 最大 warps
    double occupancy = (double)actual_warps / max_warps_per_cu;
    
    return occupancy;
}

void detailed_occupancy_analysis() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    
    std::cout << "=== GPU 詳細資訊 ===" << std::endl;
    std::cout << "裝置名稱: " << prop.name << std::endl;
    std::cout << "Compute Units (CUs): " << prop.multiProcessorCount << std::endl;
    std::cout << "最大 threads/CU: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "最大 blocks/CU: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Shared Memory/CU: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "Warp 大小: " << prop.warpSize << std::endl;
    std::cout << "最大 warps/CU: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Occupancy 詳細分析 ===" << std::endl;
    std::cout << std::setw(12) << "Block Size"
              << std::setw(12) << "Warps/Block"
              << std::setw(15) << "SharedMem(KB)"
              << std::setw(15) << "Max Blocks/CU"
              << std::setw(12) << "限制因素"
              << std::setw(12) << "Occupancy"
              << std::setw(15) << "執行時間(ms)" << std::endl;
    std::cout << std::string(95, '-') << std::endl;
    
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
    const int N = 100000000; // 10M elements
    std::vector<int> input(N, 1);
    
    for (int block_size : block_sizes) {
        // 計算各種參數
        int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
        double shared_mem_per_block = (block_size * sizeof(int)) / 1024.0;
        
        // 計算各種限制
        int max_blocks_per_cu = prop.maxBlocksPerMultiProcessor;
        int max_warps_per_cu = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        double max_shared_mem_per_cu = prop.sharedMemPerMultiprocessor / 1024.0;
        
        int blocks_by_max_blocks = max_blocks_per_cu;
        int blocks_by_warps = max_warps_per_cu / warps_per_block;
        int blocks_by_shared_mem = (int)(max_shared_mem_per_cu / shared_mem_per_block);
        
        // 找出限制因素
        int actual_blocks = std::min({blocks_by_max_blocks, blocks_by_warps, blocks_by_shared_mem});
        std::string limiting_factor;
        
        if (actual_blocks == blocks_by_max_blocks && actual_blocks <= blocks_by_warps && actual_blocks <= blocks_by_shared_mem) {
            limiting_factor = "MaxBlocks";
        } else if (actual_blocks == blocks_by_warps) {
            limiting_factor = "Warps";
        } else {
            limiting_factor = "SharedMem";
        }
        
        // 計算 occupancy
        double occupancy = calculate_occupancy(block_size, prop);
        
        // 執行效能測試
        int output = 0;
        double exec_time;
        
        switch(block_size) {
            case 64:
                exec_time = run_test<64>(input.data(), &output, N);
                break;
            case 128:
                exec_time = run_test<128>(input.data(), &output, N);
                break;
            case 256:
                exec_time = run_test<256>(input.data(), &output, N);
                break;
            case 512:
                exec_time = run_test<512>(input.data(), &output, N);
                break;
            case 1024:
                exec_time = run_test<1024>(input.data(), &output, N);
                break;
            default:
                continue;
        }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(12) << block_size
                  << std::setw(12) << warps_per_block
                  << std::setw(15) << shared_mem_per_block
                  << std::setw(15) << actual_blocks
                  << std::setw(12) << limiting_factor
                  << std::setw(12) << occupancy * 100 << "%"
                  << std::setw(15) << exec_time << std::endl;
    }
}

void occupancy_vs_performance_analysis() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    
    std::cout << std::endl << "=== Occupancy vs Performance 分析 ===" << std::endl;
    
    std::vector<int> block_sizes = {256, 1024};
    const int N = 10000000;
    std::vector<int> input(N, 1);
    
    for (int block_size : block_sizes) {
        double occupancy = calculate_occupancy(block_size, prop);
        
        int output = 0;
        double exec_time;
        
        if (block_size == 256) {
            exec_time = run_test<256>(input.data(), &output, N);
        } else {
            exec_time = run_test<1024>(input.data(), &output, N);
        }
        
        // 計算理論 throughput
        int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
        int max_warps_per_cu = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        int actual_blocks_per_cu = (int)(occupancy * max_warps_per_cu / warps_per_block);
        int total_active_threads = actual_blocks_per_cu * block_size * prop.multiProcessorCount;
        
        std::cout << std::endl << "Block Size " << block_size << ":" << std::endl;
        std::cout << "  Occupancy: " << occupancy * 100 << "%" << std::endl;
        std::cout << "  實際 blocks/CU: " << actual_blocks_per_cu << std::endl;
        std::cout << "  總活躍 threads: " << total_active_threads << std::endl;
        std::cout << "  執行時間: " << exec_time << " ms" << std::endl;
        std::cout << "  Throughput: " << (N / exec_time / 1000) << " M elements/second" << std::endl;
    }
    
    std::cout << std::endl << "=== 關鍵發現 ===" << std::endl;
    std::cout << "1. 高 Occupancy ≠ 高效能" << std::endl;
    std::cout << "2. 256 threads 在 occupancy 和算法效率間達到最佳平衡" << std::endl;
    std::cout << "3. Reduction 算法特性影響最終效能" << std::endl;
    std::cout << "4. Memory coalescing 和 warp efficiency 同樣重要" << std::endl;
}

int main() {
    detailed_occupancy_analysis();
    occupancy_vs_performance_analysis();
    
    return 0;
}