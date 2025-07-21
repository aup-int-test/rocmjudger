import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_array_to_file(filename, N, array_data):
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_1():
    N = 100
    array_data = [round(random.uniform(-100.0, 100.0), 2) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 1000
    array_data = [0.0] * N
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 4096
    array_data = [round(random.uniform(-1000.0, 1000.0), 4) for _ in range(N)]
    write_array_to_file('testcases/3.in', N, array_data)

def generate_testcase_4():
    N = 500
    array_data = [round(random.uniform(-1000.0, -0.01), 4) for _ in range(N)]
    write_array_to_file('testcases/4.in', N, array_data)

def generate_testcase_5():
    N = 9999999
    
    filename = 'testcases/5.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [round(random.uniform(-10000.0, 10000.0), 6) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 100 ===")
    generate_testcase_1()
    
    print("=== 2.in: 全0 ===")
    generate_testcase_2()
    
    print("=== 3.in: 4096 ===")
    generate_testcase_3()
    
    print("=== 4.in: 全負 ===")
    generate_testcase_4()
    
    print("=== 5.in: 9999999 ===")
    generate_testcase_5()
    
    for i in range(1, 6):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()