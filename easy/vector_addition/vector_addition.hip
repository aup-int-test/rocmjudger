#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <hip/hip_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N) C[id] = A[id] + B[id];
}

void solve(const float* A, const float* B, float* C, int N) {

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    hipDeviceSynchronize();

    hipMemcpy(C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main(int argc, char* argv[]) {
    int N;
    std::vector<float> h_A, h_B, h_C;

    // Check if file input is provided
    if (argc == 2) {
        // File input mode
        std::ifstream input_file(argv[1]);
        if (!input_file.is_open()) {
            std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
            return 1;
        }
        
        input_file >> N;
        h_A.resize(N);
        h_B.resize(N);
        h_C.resize(N);
        
        for (int i = 0; i < N; ++i) {
            input_file >> h_A[i];
        }
        for (int i = 0; i < N; ++i) {
            input_file >> h_B[i];
        }
        
        input_file.close();
    } else if (argc == 1) {
        // stdin mode
        std::cin >> N;
        h_A.resize(N);
        h_B.resize(N);
        h_C.resize(N);
        
        for (int i = 0; i < N; ++i) {
            std::cin >> h_A[i];
        }
        for (int i = 0; i < N; ++i) {
            std::cin >> h_B[i];
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " [input_file]" << std::endl;
        std::cerr << "  No arguments: Read from stdin" << std::endl;
        std::cerr << "  With file: Read from file" << std::endl;
        return 1;
    }

    solve(h_A.data(), h_B.data(), h_C.data(), N);

    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << (i + 1 < N ? ' ' : '\n');
    }

    return 0;
}
