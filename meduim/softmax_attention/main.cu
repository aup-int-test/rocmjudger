#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <float.h>

#define threadperblock 256

// nees spj

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

__global__ void computescore(const float *Q, const float *K, float *QKT, int M, int N, int d){

    // Q * KT / sqrt(d)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Q列
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K列 = KT行

    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += Q[row * d + i] * K[col * d + i];  // KT用index算就好
        }
        QKT[row * N + col] = sum / sqrtf((float)d);
    }
}

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


__global__ void computeresult(const float *QKT, const float *V, float *output, int M, int N, int d){
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // QKT列
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // V行

    if(row < M && col < d){
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += QKT[row * N + k] * V[k * d + col];
        }
        output[row * d + col] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d){

    int Qsize = M * d, Ksize = N * d, Vsize = N * d, QKTsize = M * N, outputsize = M * d;
    float *d_Q, *d_K, *d_V, *d_QKT, *d_output;

    hipMalloc(&d_Q, Qsize * sizeof(float));
    hipMalloc(&d_K, Ksize * sizeof(float));
    hipMalloc(&d_V, Vsize * sizeof(float));
    hipMalloc(&d_QKT, QKTsize * sizeof(float));
    hipMalloc(&d_output, outputsize * sizeof(float));

    hipMemcpy(d_Q, Q, Qsize * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, Ksize * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, Vsize * sizeof(float), hipMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    computescore<<<blocks, threads>>>(d_Q, d_K, d_QKT, M, N, d);
    softmax<<<M, 256>>>(d_QKT, M, N);

    blocks = dim3((d + 15) / 16, (M + 15) / 16);
    threads = dim3(16, 16);

    computeresult<<<blocks, threads>>>(d_QKT, d_V, d_output, M, N, d);
    hipMemcpy(output, d_output, outputsize * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_QKT);
    hipFree(d_output);
}

int main(){
    int M, N, d; // Q[M * d], K[N * d], V[N * d]
    std::cin >> M >> N >> d;

    std::vector<float> Q(M * d), K(N * d), V(N * d), output(M * d);

    for(int i = 0; i < M; ++i) for(int j = 0; j < d; ++j) std::cin >> Q[i * d + j];
    for(int i = 0; i < N; ++i) for(int j = 0; j < d; ++j) std::cin >> K[i * d + j]; 
    for(int i = 0; i < N; ++i) for(int j = 0; j < d; ++j) std::cin >> V[i * d + j];

    solve(Q.data(), K.data(), V.data(), output.data(), M, N, d);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < d; ++j) std::cout << output[i * d + j] << " ";
        std::cout << std::endl;
    }
}