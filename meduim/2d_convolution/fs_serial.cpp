#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

void convolution2D_cpu(const int* input, const int* kernel, int* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols,
                      int output_rows, int output_cols) {
    
    for (int row = 0; row < output_rows; row++) {
        for (int col = 0; col < output_cols; col++) {
            int sum = 0.0f;
            
            for (int kernel_r = 0; kernel_r < kernel_rows; kernel_r++) {
                for (int kernel_c = 0; kernel_c < kernel_cols; kernel_c++) {
                    int input_r = row + kernel_r;
                    int input_c = col + kernel_c;
                    
                    if (input_r < input_rows && input_c < input_cols) {
                        sum += input[input_r * input_cols + input_c] * 
                               kernel[kernel_r * kernel_cols + kernel_c];
                    }
                }
            }
            
            output[row * output_cols + col] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    std::ifstream input_file;
    std::string filename = argv[1];
    
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "File open error: " << filename << std::endl;
        return 1;
    }
    
    int input_rows, input_cols, kernel_rows, kernel_cols;
    input_file >> input_rows >> input_cols >> kernel_rows >> kernel_cols;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    std::vector<int> input(input_rows * input_cols);
    std::vector<int> kernel(kernel_rows * kernel_cols);
    std::vector<int> output(output_rows * output_cols);

    for(int i = 0; i < input_rows; ++i) {
        for(int j = 0; j < input_cols; ++j) {
            input_file >> input[i * input_cols + j];
        }
    }
    
    for(int i = 0; i < kernel_rows; ++i) {
        for(int j = 0; j < kernel_cols; ++j) {
            input_file >> kernel[i * kernel_cols + j];
        }
    }

    input_file.close();

    convolution2D_cpu(input.data(), kernel.data(), output.data(), 
                     input_rows, input_cols, kernel_rows, kernel_cols,
                     output_rows, output_cols);

    //std::cout << std::fixed << std::setprecision(1);
    for(int i = 0; i < output_rows; ++i) {
        for(int j = 0; j < output_cols; ++j) {
            std::cout << output[i * output_cols + j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}