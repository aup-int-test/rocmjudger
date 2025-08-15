import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_convolution_to_file(filename, input_data, kernel_data):
    with open(filename, 'w') as f:
        f.write(f"{len(input_data)} {len(kernel_data)}\n")
        f.write(" ".join(map(str, input_data)) + "\n")
        f.write(" ".join(map(str, kernel_data)) + "\n")

def generate_testcase_1():
    size = 100
    input_data = [round(random.uniform(-10.0, 10.0), 2) for _ in range(size)]
    kernel_data = [round(random.uniform(-1.0, 1.0), 2) for _ in range(size)]
    write_convolution_to_file('testcases/1.in', input_data, kernel_data)

def generate_testcase_2():
    input_size = 10000
    kernel_size = 1
    input_data = [round(random.uniform(-100.0, 100.0), 2) for _ in range(input_size)]
    kernel_data = [round(random.uniform(-1.0, 1.0), 2) for _ in range(kernel_size)]
    write_convolution_to_file('testcases/2.in', input_data, kernel_data)

def generate_testcase_3():
    input_size = 50000
    kernel_size = 1000
    input_data = [0.0] * input_size
    kernel_data = [0.0] * kernel_size
    write_convolution_to_file('testcases/3.in', input_data, kernel_data)

def generate_testcase_4():
    input_size = 100000
    kernel_size = 5000
    input_data = [round(random.uniform(-1000.0, 1000.0), 2) for _ in range(input_size)]
    kernel_data = [round(random.uniform(-10.0, 10.0), 2) for _ in range(kernel_size)]
    write_convolution_to_file('testcases/4.in', input_data, kernel_data)

def generate_testcase_5():
    input_size = 1000000
    kernel_size = 10
    
    filename = 'testcases/5.in'
    with open(filename, 'w') as f:
        f.write(f"{input_size} {kernel_size}\n")
        
        input_data = [round(random.uniform(-999999.0, 999999.0), 6) for _ in range(input_size)]
        f.write(" ".join(map(str, input_data)) + "\n")
        
        kernel_data = [round(random.uniform(-1.0, 1.0), 6) for _ in range(kernel_size)]
        f.write(" ".join(map(str, kernel_data)) + "\n")

def generate_testcase_6():
    input_size = 1000000
    kernel_size = 1000
    
    filename = 'testcases/6.in'
    with open(filename, 'w') as f:
        f.write(f"{input_size} {kernel_size}\n")
        
        input_data = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(input_size)]
        f.write(" ".join(map(str, input_data)) + "\n")
        
        kernel_data = [round(random.uniform(-10.0, 10.0), 6) for _ in range(kernel_size)]
        f.write(" ".join(map(str, kernel_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 100個，input_size == kernel_size ===")
    generate_testcase_1()

    print("=== 2.in: 10000，kernel_size = 1 ===")
    generate_testcase_2()

    print("=== 3.in: 50000個0 ===")
    generate_testcase_3()

    print("=== 4.in: 100000 ===")
    generate_testcase_4()

    print("=== 5.in: 1000000 ===")
    generate_testcase_5()

    print("=== 6.in: 1000000 ===")
    generate_testcase_6()
    
    print("done!")
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()