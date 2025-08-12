#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <stdio.h>

#include <fstream>

template <int blk_size>
__global__ void scan_kernel(const int* input, int* output, int* rst_next_level, int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * blk_size * 2;

    __shared__ int temp[blk_size << 1];

    if(offset + tid < N) 
        temp[tid] = input[offset + tid];
    else 
        temp[tid] = 0;

    if(offset + tid + blk_size < N) 
        temp[tid + blk_size] = input[offset + tid + blk_size];
    else 
        temp[tid + blk_size] = 0;

    __syncthreads();

    for(int step = 1; step < (blk_size << 1); step *= 2) {
        int read_pos = tid * 2 + 1;  
        
        if(read_pos < (blk_size << 1) && read_pos >= step) {
            temp[read_pos] += temp[read_pos - step];
        }
        
        read_pos = tid * 2 + 2;   
        if(read_pos < (blk_size << 1) && read_pos >= step) {
            temp[read_pos] += temp[read_pos - step];
        }
        
        __syncthreads();
    }
    if(offset + tid < N) 
        output[offset + tid] = temp[tid];
    if(offset + tid + blk_size < N) 
        output[offset + tid + blk_size] = temp[tid + blk_size];
    
    if(tid == 0) 
        rst_next_level[bid] = temp[(blk_size << 1) - 1];
}

__global__ void scan_kernel_serial(int* input_output, int N) {
    for(int i = 1; i < N; i++)  input_output[i] += input_output[i - 1];
}

__global__ void walk_back_kernel(const int blk_size_2, int* input_output, int* rst_level_1, int* rst_level_2, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = bid * blockDim.x + tid;
    const int bid_2 = bid / blk_size_2;
    const int add_1 = (bid & (blk_size_2 - 1)) ? rst_level_1[bid - 1] : 0.0f;
    const int add_2 = bid_2 ? rst_level_2[bid_2 - 1] : 0.0f;
    
    if(global_id < N) input_output[global_id] += add_1 + add_2;
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N) {
    
    int *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int));
    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);

    const int BLOCK_SIZE_1 = 256;
    const int BLOCK_ELEMENTS_1 = BLOCK_SIZE_1 << 1;
    const int BLOCK_SIZE_2 = 64;
    const int BLOCK_ELEMENTS_2 = BLOCK_SIZE_2 << 1;
    int blk_num_1s = (N + BLOCK_ELEMENTS_1 - 1) / BLOCK_ELEMENTS_1;
    int blk_num_2s = (blk_num_1s + BLOCK_ELEMENTS_2 - 1) / BLOCK_ELEMENTS_2;
    int* d_output_1s, *d_output_2s;
    hipMalloc(&d_output_1s, blk_num_1s * sizeof(int));
    hipMalloc(&d_output_2s, blk_num_2s * sizeof(int));

    // first level scan
    scan_kernel<BLOCK_SIZE_1><<<blk_num_1s, BLOCK_SIZE_1>>>(d_input, d_output, d_output_1s, N);
    hipDeviceSynchronize();

    // second level scan
    scan_kernel<BLOCK_SIZE_2><<<blk_num_2s, BLOCK_SIZE_2>>>(d_output_1s, d_output_1s, d_output_2s, blk_num_1s);
    hipDeviceSynchronize();

    // third level scan
    scan_kernel_serial<<<1, 1>>>(d_output_2s, blk_num_2s);
    hipDeviceSynchronize();

    // walk back
    walk_back_kernel<<<blk_num_1s, BLOCK_ELEMENTS_1>>>(BLOCK_ELEMENTS_2, d_output, d_output_1s, d_output_2s, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, N * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_output_1s);
    hipFree(d_output_2s);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    std::ifstream input_file;
    std::string filename = argv[1];
    
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }
    int N;
    input_file >> N;

    std::vector<int> input(N), output(N);

    for(int i = 0; i < N; ++i) input_file >> input[i];

    input_file.close();

    solve(input.data(), output.data(), N);

    for(int i = 0; i < N; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
}