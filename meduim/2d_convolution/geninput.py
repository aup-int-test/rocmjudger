import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_conv2d_to_file(filename, input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data):
    with open(filename, 'w') as f:
        f.write(f"{input_rows} {input_cols} {kernel_rows} {kernel_cols}\n")
        for i in range(input_rows):
            for j in range(input_cols):
                f.write(f"{input_data[i * input_cols + j]} ")
            f.write("\n")
        for i in range(kernel_rows):
            for j in range(kernel_cols):
                f.write(f"{kernel_data[i * kernel_cols + j]} ")
            f.write("\n")

def generate_testcase_1():
    input_rows, input_cols = 64, 64
    kernel_rows, kernel_cols = 3, 3
    input_data = [random.randint(-10, 10) for _ in range(input_rows * input_cols)]
    kernel_data = [random.randint(-5, 5) for _ in range(kernel_rows * kernel_cols)]
    write_conv2d_to_file('testcases/1.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def generate_testcase_2():
    input_rows, input_cols = 512, 512
    kernel_rows, kernel_cols = 5, 5
    input_data = [random.randint(-50, 50) for _ in range(input_rows * input_cols)]
    kernel_data = [random.randint(-10, 10) for _ in range(kernel_rows * kernel_cols)]
    write_conv2d_to_file('testcases/2.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def generate_testcase_3():
    input_rows, input_cols = 2048, 2048
    kernel_rows, kernel_cols = 7, 7
    input_data = [random.randint(-100, 100) for _ in range(input_rows * input_cols)]
    kernel_data = [random.randint(-20, 20) for _ in range(kernel_rows * kernel_cols)]
    write_conv2d_to_file('testcases/3.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def generate_testcase_4():
    input_rows, input_cols = 1, 1
    kernel_rows, kernel_cols = 1, 1
    input_data = [random.randint(-10, 10)]
    kernel_data = [random.randint(-5, 5)]
    write_conv2d_to_file('testcases/4.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def generate_testcase_5():
    input_rows, input_cols = 17, 33
    kernel_rows, kernel_cols = 17, 33
    input_data = [random.randint(-50, 50) for _ in range(input_rows * input_cols)]
    kernel_data = [random.randint(-10, 10) for _ in range(kernel_rows * kernel_cols)]
    write_conv2d_to_file('testcases/5.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def generate_testcase_6():
    input_rows, input_cols = 1023, 1025
    kernel_rows, kernel_cols = 31, 29
    input_data = [random.randint(-100, 100) for _ in range(input_rows * input_cols)]
    kernel_data = [random.randint(-50, 50) for _ in range(kernel_rows * kernel_cols)]
    write_conv2d_to_file('testcases/6.in', input_rows, input_cols, kernel_rows, kernel_cols, input_data, kernel_data)

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 小測資 64x64 ===")
    generate_testcase_1()
    
    print("=== 2.in: 中測資 512x512 ===")
    generate_testcase_2()
    
    print("=== 3.in: 大測資 2048x2048 ===")
    generate_testcase_3()
    
    print("=== 4.in: Edge case 最小尺寸 1x1 ===")
    generate_testcase_4()
    
    print("=== 5.in: Edge case 輸入等於核 17x33 ===")
    generate_testcase_5()
    
    print("=== 6.in: Edge case 不對齊block size 1023x1025 ===")
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()