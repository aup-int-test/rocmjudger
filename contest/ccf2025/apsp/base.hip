#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define DEV_NO 0

const int INF = ((1 << 30) - 1);
const int B = 64;
const int n_thread = 32;
const int n_per_iter = 4;
const int n_sm_size = 4096;
const int shift = 6;

void input(char* infile);
void output(char* outFileName);
void block_FW();

__global__ void cal1(int *D, int r, int n);
__global__ void cal2(int *D, int r, int n);
__global__ void cal3(int *D, int r, int n);

int n, m, padding_n;
int *Dist, *dev_Dist;
size_t Dist_size;

int main(int argc, char* argv[]){
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }
    hipSetDevice(DEV_NO);

    input(argv[1]);
    block_FW();
    output(argv[2]);

    if (dev_Dist) hipFree(dev_Dist);
    if (Dist)     hipHostFree(Dist);
    return 0;
}

void input(char* infile){
    FILE* file = fopen(infile, "rb");
    if (!file) { perror("fopen input"); exit(1); }

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    padding_n = n + B - ((n % B + B - 1) % B + 1);
    Dist_size = (size_t)padding_n * (size_t)padding_n * sizeof(int);

    if (hipHostMalloc(&Dist, Dist_size) != hipSuccess) {
        fprintf(stderr, "hipHostMalloc failed\n"); exit(1);
    }

    for(int i = 0; i < padding_n; i++){
        for(int j = 0; j < padding_n; j++){
            if((i == j) && (i < n)) {
                Dist[i * padding_n + j] = 0;
            }else{
                Dist[i * padding_n + j] = INF;
            }
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * padding_n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName){
    FILE* outfile = fopen(outFileName, "wb"); 
    if (!outfile) { perror("fopen output"); exit(1); }
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i * padding_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void block_FW(){
    int round = (padding_n + B - 1) / B;
    dim3 block(n_thread, n_thread), grid2(2, round - 1), grid3(round - 1, round - 1);

    if (hipMalloc(&dev_Dist, Dist_size) != hipSuccess) {
        fprintf(stderr, "hipMalloc failed\n"); exit(1);
    }
    hipMemcpy(dev_Dist, Dist, Dist_size, hipMemcpyHostToDevice);

    for(int r = 0; r < round; r++){
        hipLaunchKernelGGL(cal1, dim3(1), block, 0, 0, dev_Dist, r, padding_n);
        hipLaunchKernelGGL(cal2, grid2,  block, 0, 0, dev_Dist, r, padding_n);
        hipLaunchKernelGGL(cal3, grid3,  block, 0, 0, dev_Dist, r, padding_n);
    }

    hipMemcpy(Dist, dev_Dist, Dist_size, hipMemcpyDeviceToHost);
    hipFree(dev_Dist);  
}

__global__ void cal1(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; // 64 * 64

    int b_i, b_j;
    int i, j;
    int ij, ik, kj;

    b_i = b_j = r << shift;
    i = threadIdx.y; j = threadIdx.x; // < n_thread
    ij = (threadIdx.y * n_thread + threadIdx.x) * n_per_iter;

    sm[ij]     = D[(b_i + i)             * n + b_j + j];
    sm[ij + 1] = D[(b_i + i)             * n + b_j + j + n_thread];
    sm[ij + 2] = D[(b_i + i + n_thread)  * n + b_j + j];
    sm[ij + 3] = D[(b_i + i + n_thread)  * n + b_j + j + n_thread];

    #pragma unroll 32
    for(int k = 0; k < n_thread; k++){
        __syncthreads(); 
        ik = (i * n_thread + k) * 4;
        kj = (k * n_thread + j) * 4;

        sm[ij]     = min(sm[ij],     sm[ik]     + sm[kj]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik]     + sm[kj + 1]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 2] + sm[kj]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 2] + sm[kj + 1]);

        sm[ij]     = min(sm[ij],     sm[ik + 1] + sm[kj + 2]);
        sm[ij + 1] = min(sm[ij + 1], sm[ik + 1] + sm[kj + 3]);
        sm[ij + 2] = min(sm[ij + 2], sm[ik + 3] + sm[kj + 2]);
        sm[ij + 3] = min(sm[ij + 3], sm[ik + 3] + sm[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j]                   = sm[ij];
    D[(b_i + i) * n + b_j + j + n_thread]        = sm[ij + 1];
    D[(b_i + i + n_thread) * n + b_j + j]        = sm[ij + 2];
    D[(b_i + i + n_thread) * n + b_j + j + n_thread] = sm[ij + 3];
}

__global__ void cal2(int *D, int r, int n){
    __shared__ int sm[n_sm_size]; // 64 * 64
    __shared__ int cp[n_sm_size]; 

    int b_i, b_j, b_k;
    int i, j;
    volatile int ik, kj;
    int tmp[4];

    b_i = (blockIdx.x * r + (!blockIdx.x) * (blockIdx.y + (blockIdx.y >= r))) << shift;
    b_j = (blockIdx.x * (blockIdx.y + (blockIdx.y >= r)) + (!blockIdx.x) * r) << shift;
    b_k = r << shift;
    i = threadIdx.y; j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    // local
    tmp[0] = D[(b_i + i) * n + b_j + j];
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    // ik tile
    sm[ik]     = D[(b_i + i)             * n + b_k + j];
    sm[ik + 1] = D[(b_i + i)             * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread)  * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread)  * n + b_k + j + n_thread];

    // kj tile
    cp[kj]     = D[(b_k + i)             * n + b_j + j];
    cp[kj + 1] = D[(b_k + i)             * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread)  * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread)  * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll 32
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;

        tmp[0] = min(min(tmp[0], sm[ik]     + cp[kj]),     sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik]     + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]),     sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j]                        = tmp[0];
    D[(b_i + i) * n + b_j + j + n_thread]             = tmp[1];
    D[(b_i + i + n_thread) * n + b_j + j]             = tmp[2];
    D[(b_i + i + n_thread) * n + b_j + j + n_thread]  = tmp[3];
}

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
    i = threadIdx.y; j = threadIdx.x;
    ik = kj = (i * n_thread + j) * n_per_iter;

    tmp[0] = D[(b_i + i) * n + b_j + j];
    tmp[1] = D[(b_i + i) * n + b_j + j + n_thread];
    tmp[2] = D[(b_i + i + n_thread) * n + b_j + j];
    tmp[3] = D[(b_i + i + n_thread) * n + b_j + j + n_thread];

    sm[ik]     = D[(b_i + i)             * n + b_k + j];
    sm[ik + 1] = D[(b_i + i)             * n + b_k + j + n_thread];
    sm[ik + 2] = D[(b_i + i + n_thread)  * n + b_k + j];
    sm[ik + 3] = D[(b_i + i + n_thread)  * n + b_k + j + n_thread];

    cp[kj]     = D[(b_k + i)             * n + b_j + j];
    cp[kj + 1] = D[(b_k + i)             * n + b_j + j + n_thread];
    cp[kj + 2] = D[(b_k + i + n_thread)  * n + b_j + j];
    cp[kj + 3] = D[(b_k + i + n_thread)  * n + b_j + j + n_thread];

    __syncthreads();

    #pragma unroll 32
    for(int k = 0; k < n_thread; k++){
        ik = (i * n_thread + k) * n_per_iter;
        kj = (k * n_thread + j) * n_per_iter;

        tmp[0] = min(min(tmp[0], sm[ik]     + cp[kj]),     sm[ik + 1] + cp[kj + 2]);
        tmp[1] = min(min(tmp[1], sm[ik]     + cp[kj + 1]), sm[ik + 1] + cp[kj + 3]);
        tmp[2] = min(min(tmp[2], sm[ik + 2] + cp[kj]),     sm[ik + 3] + cp[kj + 2]);
        tmp[3] = min(min(tmp[3], sm[ik + 2] + cp[kj + 1]), sm[ik + 3] + cp[kj + 3]);
    }

    D[(b_i + i) * n + b_j + j]                        = tmp[0];
    D[(b_i + i) * n + b_j + j + n_thread]             = tmp[1];
    D[(b_i + i + n_thread) * n + b_j + j]             = tmp[2];
    D[(b_i + i + n_thread) * n + b_j + j + n_thread]  = tmp[3];
}
