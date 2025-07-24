#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <float.h>

void computescore(const float *Q, const float *K, float *QKT, int M, int N, int d) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += Q[row * d + i] * K[col * d + i];
            }
            QKT[row * N + col] = sum / sqrtf((float)d);
        }
    }
}

void softmax(float *QKT, int M, int N) {
    for (int row = 0; row < M; row++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            max_val = fmaxf(max_val, QKT[row * N + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            QKT[row * N + j] = expf(QKT[row * N + j] - max_val);
            sum_exp += QKT[row * N + j];
        }
        
        for (int j = 0; j < N; j++) {
            QKT[row * N + j] /= sum_exp;
        }
    }
}

void computeresult(const float *QKT, const float *V, float *output, int M, int N, int d) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < d; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += QKT[row * N + k] * V[k * d + col];
            }
            output[row * d + col] = sum;
        }
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    std::vector<float> QKT(M * N);
    
    computescore(Q, K, QKT.data(), M, N, d);
    softmax(QKT.data(), M, N);
    computeresult(QKT.data(), V, output, M, N, d);
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
    
    int M, N, d;
    input_file >> M >> N >> d;

    std::vector<float> Q(M * d), K(N * d), V(N * d), output(M * d);

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