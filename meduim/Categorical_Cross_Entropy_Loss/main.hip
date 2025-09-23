#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

/*sample input
3 4 
-0.5 1.5 0.0 1.0 
2.0 -1.0 0.5 0.5 
0.0 0.0 0.0 0.0
3 0 1

0.988204

- - -

2 3
1.0 2.0 0.5
0.1 3.0 1.5
1 1

0.354893
*/


__global__ void loss_kernel(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j < N) {
        float exp_sum = 0.0f;
        for(int k = 0; k < C; ++k) {
            exp_sum += expf(logits[j*C+k]);
        }
        const float loss_j = logf(exp_sum) - logits[j*C+true_labels[j]];
        atomicAdd(loss,loss_j);
    }
}
__global__ void div_kernel(float* a, int b) {
    *a /= b;
}

void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {

    float *d_logits, *d_loss;
    int *d_true_labels;

    hipMalloc(&d_logits, N * C * sizeof(float));
    hipMalloc(&d_true_labels, N * sizeof(int));
    hipMalloc(&d_loss, sizeof(float));

    hipMemcpy(d_logits, logits, N * C * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_true_labels, true_labels, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_loss, loss, sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    loss_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_logits, d_true_labels, d_loss, N, C);
    div_kernel<<<1,1>>>(d_loss, N);
    hipDeviceSynchronize();

    hipMemcpy(loss, d_loss, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_logits);
    hipFree(d_true_labels);
    hipFree(d_loss);
}



int main(){

    int N, C;

    std::cin >> N >> C;

    std::vector<float> logits(N * C);
    std::vector<int> true_labels(N);
    float loss;

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < C; ++j) std::cin >> logits[i * C + j];
    }

    for(int i = 0; i < N; ++i) std::cin >> true_labels[i];

    solve(logits.data(), true_labels.data(), &loss, N, C);

    std::cout << std::fixed << std::setprecision(6) << loss << std::endl;
}