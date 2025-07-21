#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>

#include <fstream>

#define threadperblock 256

__global__ void reduction(const float* input, float* output, int N){
    __shared__ float sdata[threadperblock];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    sdata[tidx] = (idx < N) ? input[idx] : 0.0f;

    __syncthreads();

    // Parallel reduction within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) sdata[tidx] += sdata[tidx + i];
        __syncthreads();
    }

    if (tidx == 0) atomicAdd(output, sdata[0]); 
}


extern "C" void solve(const float* input, float* output, int N) {  

    float *d_input, *d_output;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));

    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    int blockpergrid = (N + threadperblock - 1) / threadperblock;

    //size_t shared_mem_size = threadperblock * sizeof(float);
    reduction<<<blockpergrid, threadperblock>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
}

int main(int argc, char* argv[]){
    std::ifstream infile;
    std::ofstream outfile;
    
    // 決定輸入來源
    if (argc > 1) {
        infile.open(argv[1]);
        if (!infile.is_open()) {
            std::cerr << "Error: Cannot open input file " << argv[1] << std::endl;
            return 1;
        }
    }
    
    // 決定輸出目標
    if (argc > 2) {
        outfile.open(argv[2]);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open output file " << argv[2] << std::endl;
            return 1;
        }
    }
    
    // 選擇輸入流
    std::istream& input_stream = (argc > 1) ? infile : std::cin;
    std::ostream& output_stream = (argc > 2) ? outfile : std::cout;
    
    int N;
    float output;
    
    input_stream >> N;
    std::vector<float> input(N);

    for(int i = 0; i < N; ++i) {
        input_stream >> input[i];
    }

    solve(input.data(), &output, N);

    output_stream << output << std::endl;
    
    // 關閉檔案
    if (infile.is_open()) infile.close();
    if (outfile.is_open()) outfile.close();
    
    return 0;
}