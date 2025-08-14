#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <fstream>
#define DEV_NO 0

const int INF = ((1 << 30) - 1);
const int B = 64;
const int n_thread = 32;
const int n_per_iter = 4;
const int n_sm_size = 4096;
const int shift = 6;

void block_FW();

__global__ void cal1(int *D, int r, int n);
__global__ void cal2(int *D, int r, int n);
__global__ void cal3(int *D, int r, int n);

int n, m, padding_n;
std::vector<int> Dist;  
int *d_Dist;          
size_t Dist_size;

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error " << argv[1] << std::endl;
        return 1;
    }

    input_file >> n >> m;
    padding_n = ((n + B - 1) / B) * B;
    Dist_size = padding_n * padding_n * sizeof(int);

    Dist.resize(padding_n * padding_n, INF);
    for(int i = 0; i < n; i++){
        Dist[i * padding_n + i] = 0;
    }

    int u, v, w;
    for(int i = 0; i < m; i++){
        input_file >> u >> v >> w;
        Dist[u * padding_n + v] = w;
    }
    input_file.close();

    block_FW();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j) std::cout << ' ';
            std::cout << Dist[i * padding_n + j];
        }
        std::cout << "\n";
    }

    return 0;
}

void block_FW(){
    int round = (padding_n + B - 1) / B;
    dim3 block(n_thread, n_thread), grid2(2, round - 1), grid3(round - 1, round - 1);

    hipMalloc(&d_Dist, Dist_size);
    hipMemcpy(d_Dist, Dist.data(), Dist_size, hipMemcpyHostToDevice);

    for(int r = 0; r < round; r++){
        cal1<<<1, block>>>(d_Dist, r, padding_n);
        cal2<<<grid2, block>>>(d_Dist, r, padding_n);
        cal3<<<grid3, block>>>(d_Dist, r, padding_n);
    }

    hipMemcpy(Dist.data(), d_Dist, Dist_size, hipMemcpyDeviceToHost);
    hipFree(d_Dist);
}

__global__ void cal1(int *D, int r, int n){
    __shared__ int sm[n_sm_size];
    int b_i, b_j;
    int i, j;
    int ij, ik, kj;
    
    b_i = b_j = r << shift; 
    i = threadIdx.y; j = threadIdx.x; 
    ij = (threadIdx.y * n_thread + threadIdx.x) * n_per_iter;

    sm[ij] = D[(b_i + i) * n + b_j + j]; 
    sm[ij + 1] = D[(b_i + i) * n + b_j + j + n_thread];
    sm[ij + 2] = D[(b_i + i + n_thread) * n + b_j + j];
    sm[ij + 3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        __syncthreads(); 
        ik = (i * n_thread + k) * 4;
        kj = (k * n_thread + j) * 4;
        sm[ij] = min(sm[ij], sm[ik] + sm[kj]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik] + sm[kj + 1]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 2] + sm[kj]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 2] + sm[kj + 1]);

        sm[ij] = min(sm[ij], sm[ik + 1] + sm[kj + 2]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik + 1] + sm[kj + 3]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 3] + sm[kj + 2]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 3] + sm[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = sm[ij]; 
    D[(b_i + i) * n + b_j + j + n_thread] = sm[ij + 1]; 
    D[(b_i + i + n_thread) * n + b_j + j] = sm[ij + 2]; 
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = sm[ij + 3]; 
}

__global__ void cal2(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; 
    __shared__ int cp[n_sm_size]; 
    
    int b_i, b_j, b_k;
    int i, j;
    volatile int ik, kj;
    int tmp[4];
    
    b_i = (blockIdx.x * r + (!blockIdx.x) * (blockIdx.y + (blockIdx.y >= r))) << shift;
    b_j = (blockIdx.x * (blockIdx.y + (blockIdx.y >= r)) + (!blockIdx.x) * r) << shift;
    b_k = r << shift;
    i = threadIdx.y, j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    tmp[0] = D[(b_i + i) * n + b_j + j]; 
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    sm[ik] = D[(b_i + i) * n + b_k + j]; 
    sm[ik + 1] = D[(b_i + i) * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread) * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread) * n + b_k + j + n_thread];

    cp[kj] = D[(b_k + i) * n + b_j + j]; 
    cp[kj + 1] = D[(b_k + i) * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread) * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread) * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;
        tmp[0] = min(min(tmp[0], sm[ik] + cp[kj]), sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik] + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]), sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = tmp[0]; 
    D[(b_i + i) * n + b_j + j + n_thread] = tmp[1]; 
    D[(b_i + i + n_thread) * n + b_j + j] = tmp[2]; 
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = tmp[3]; 
};

__global__ void cal3(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; 
    __shared__ int cp[n_sm_size]; 
    
    int b_i, b_j, b_k;
    int i, j;
    volatile int ik, kj;
    int tmp[4];

    b_i = (blockIdx.x + (blockIdx.x >= r)) << shift;
    b_j = (blockIdx.y + (blockIdx.y >= r)) << shift;
    b_k = r << shift;
    i = threadIdx.y, j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    tmp[0] = D[(b_i + i) * n + b_j + j]; 
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    sm[ik] = D[(b_i + i) * n + b_k + j]; 
    sm[ik + 1] = D[(b_i + i) * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread) * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread) * n + b_k + j + n_thread];

    cp[kj] = D[(b_k + i) * n + b_j + j]; 
    cp[kj + 1] = D[(b_k + i) * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread) * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread) * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll n_thread
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;
        tmp[0] = min(min(tmp[0], sm[ik] + cp[kj]), sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik] + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]), sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j] = tmp[0];    
    D[(b_i + i) * n + b_j + j + n_thread] = tmp[1];   
    D[(b_i + i + n_thread) * n + b_j + j] = tmp[2];   
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = tmp[3];    
};