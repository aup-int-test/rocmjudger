#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin.is_open()) {
        std::cerr << "fileopen error " << argv[1] << "\n";
        return 1;
    }

    int N, C;
    if (!(fin >> N >> C) || N <= 0 || C <= 0) {
        std::cerr << "bad header\n";
        return 1;
    }

    std::vector<float> logits(N * C);
    std::vector<int>   labels(N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < C; ++j)
            fin >> logits[i * C + j];

    for (int i = 0; i < N; ++i)
        fin >> labels[i];

    if (!fin) {
        std::cerr << "bad input values\n";
        return 1;
    }

    // match HIP version: naive log-sum-exp (no max-trick), float math
    float total_loss = 0.0f;
    for (int i = 0; i < N; ++i) {
        float exp_sum = 0.0f;
        const float* row = &logits[i * C];
        for (int k = 0; k < C; ++k) {
            exp_sum += std::exp(row[k]);   // expf
        }
        int y = labels[i];
        // assume 0 <= y < C (same as GPU code; no bounds check)
        float loss_i = std::log(exp_sum) - row[y];  // logf
        total_loss += loss_i;
    }

    float avg = total_loss / static_cast<float>(N);
    std::cout << std::fixed << std::setprecision(6) << avg << "\n";
    return 0;
}
