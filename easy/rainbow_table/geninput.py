import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_hash_to_file(filename, N, R, input_data):
    with open(filename, 'w') as f:
        f.write(f"{N} {R}\n")
        f.write(" ".join(map(str, input_data)) + "\n")

def generate_testcase_1():
    N, R = 10, 1
    input_data = [random.randint(-1000, 1000) for _ in range(N)]
    write_hash_to_file('testcases/1.in', N, R, input_data)

def generate_testcase_2():
    N, R = 100, 5
    input_data = [random.randint(-100000, 100000) for _ in range(N)]
    write_hash_to_file('testcases/2.in', N, R, input_data)

def generate_testcase_3():
    N, R = 1000, 10
    input_data = [0] * N
    write_hash_to_file('testcases/3.in', N, R, input_data)

def generate_testcase_4():
    N, R = 5000, 1
    input_data = [random.randint(-2147483648, 2147483647) for _ in range(N)]
    write_hash_to_file('testcases/4.in', N, R, input_data)

def generate_testcase_5():
    N, R = 10000, 100
    input_data = [random.randint(1, 1000000) for _ in range(N)]
    write_hash_to_file('testcases/5.in', N, R, input_data)

def generate_testcase_6():
    N, R = 1000000, 100
    
    filename = 'testcases/6.in'
    with open(filename, 'w') as f:
        f.write(f"{N} {R}\n")
        
        input_data = [random.randint(-1000000, 1000000) for _ in range(N)]
        f.write(" ".join(map(str, input_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 小陣列單次Hash ===")
    generate_testcase_1()
    
    print("=== 2.in: 中等陣列多次Hash ===")
    generate_testcase_2()
    
    print("=== 3.in: 全0陣列 ===")
    generate_testcase_3()
    
    print("=== 4.in: 大範圍整數 ===")
    generate_testcase_4()
    
    print("=== 5.in: 高迭代次數 ===")
    generate_testcase_5()
    
    print("=== 6.in: 大陣列、高迭代次數 ===")
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()