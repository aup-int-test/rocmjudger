#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <limits.h>

#include <fstream>

#define threadperblock 256

/*basic testcase
input:
2 3 4
1. 0. 0. 0.
0. 1. 0. 0.
1. 0. 0. 0.
0. 1. 0. 0.
0. 0. 1. 0.
1. 2. 3. 4.
5. 6. 7. 8.
9. 10. 11. 12.

output:
4.29 5.29 6.29 7.29
5. 6. 7. 8.
*/

__global__ void computescore(const int *Q, const int *K, int *QKT, int M, int N, int d){

    // Q * KT / sqrt(d)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Q列
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K列 = KT行

    if (row < M && col < N){
        int sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += Q[row * d + i] * K[col * d + i];  // KT用index算就好
        }
        QKT[row * N + col] = sum / sqrtf((int)d);
    }
}

__global__ void softmax(int *QKT, int M, int N) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (row < M) {
        int max_val = -INT_MAX;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, QKT[row * N + j]);
        }
        
        int sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            QKT[row * N + j] = expf(QKT[row * N + j] - max_val);
            sum_exp += QKT[row * N + j];
        }
        
        for (int j = 0; j < N; ++j) {
            QKT[row * N + j] /= sum_exp;
        }
    }
}


__global__ void computeresult(const int *QKT, const int *V, int *output, int M, int N, int d){
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // QKT列
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // V行

    if(row < M && col < d){
        int sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += QKT[row * N + k] * V[k * d + col];
        }
        output[row * d + col] = sum;
    }
}

extern "C" void solve(const int* Q, const int* K, const int* V, int* output, int M, int N, int d){

    int Qsize = M * d, Ksize = N * d, Vsize = N * d, QKTsize = M * N, outputsize = M * d;
    int *d_Q, *d_K, *d_V, *d_QKT, *d_output;

    hipMalloc(&d_Q, Qsize * sizeof(int));
    hipMalloc(&d_K, Ksize * sizeof(int));
    hipMalloc(&d_V, Vsize * sizeof(int));
    hipMalloc(&d_QKT, QKTsize * sizeof(int));
    hipMalloc(&d_output, outputsize * sizeof(int));

    hipMemcpy(d_Q, Q, Qsize * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, Ksize * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, Vsize * sizeof(int), hipMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    computescore<<<blocks, threads>>>(d_Q, d_K, d_QKT, M, N, d);
    softmax<<<M, 256>>>(d_QKT, M, N);

    blocks = dim3((d + 15) / 16, (M + 15) / 16);
    threads = dim3(16, 16);

    computeresult<<<blocks, threads>>>(d_QKT, d_V, d_output, M, N, d);
    hipMemcpy(output, d_output, outputsize * sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_QKT);
    hipFree(d_output);
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
    int M, N, d; // Q[M * d], K[N * d], V[N * d]
    input_file >> M >> N >> d;

    std::vector<int> Q(M * d), K(N * d), V(N * d), output(M * d);

    for(int i = 0; i < M; ++i) for(int j = 0; j < d; ++j) input_file >> Q[i * d + j];
    for(int i = 0; i < N; ++i) for(int j = 0; j < d; ++j) input_file >> K[i * d + j]; 
    for(int i = 0; i < N; ++i) for(int j = 0; j < d; ++j) input_file >> V[i * d + j];

    input_file.close();

    solve(Q.data(), K.data(), V.data(), output.data(), M, N, d);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < d; ++j) std::cout << output[i * d + j] << " ";
        std::cout << std::endl;
    }
}