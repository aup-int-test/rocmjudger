#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

void serial_histogram(const int* input, int* histogram, int N, int num_bins) {

    for (int i = 0; i < num_bins; i++) {
        histogram[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        histogram[input[i]]++;
    }
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
        std::cerr << "fileopen error: " << filename << std::endl;
        return 1;
    }
    
    int N, num_bins;
    input_file >> N >> num_bins;

    std::vector<int> input(N);
    std::vector<int> histogram(num_bins);
    
    for (int i = 0; i < N; ++i) {
        input_file >> input[i];
    }
    
    input_file.close();

    serial_histogram(input.data(), histogram.data(), N, num_bins);
    
    for (int i = 0; i < num_bins; ++i) {
        std::cout << histogram[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}