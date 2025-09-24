#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in.is_open()) {
        std::cerr << "fileopen error: " << argv[1] << "\n";
        return 1;
    }

    int N;
    in >> N;
    if (!in || N <= 0) {
        std::cerr << "Invalid N\n";
        return 1;
    }

    // 讀入 predictions
    std::vector<float> predictions(N);
    for (int i = 0; i < N; ++i) {
        if (!(in >> predictions[i])) {
            std::cerr << "Bad input on predictions at i=" << i << "\n";
            return 1;
        }
    }

    long long n_read_targets = 0;
    double sum = 0.0;
    double c = 0.0; // compensation

    for (int i = 0; i < N; ++i) {
        float t;
        if (!(in >> t)) {
            std::cerr << "Bad input on targets at i=" << i << "\n";
            return 1;
        }
        ++n_read_targets;

        double diff = static_cast<double>(predictions[i]) - static_cast<double>(t);
        double term = diff * diff;
        double tsum = sum + term;
        if (std::fabs(sum) >= std::fabs(term)) {
            c += (sum - tsum) + term; 
        } else {
            c += (term - tsum) + sum;
        }
        sum = tsum;
    }

    if (n_read_targets != N) {
        std::cerr << "Targets count mismatch\n";
        return 1;
    }

    double ssd = sum + c;                 
    double mse = ssd / static_cast<double>(N);

    std::cout << std::fixed << std::setprecision(6)
              << static_cast<float>(mse) << "\n";
    return 0;
}
