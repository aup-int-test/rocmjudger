import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_matrix_to_file(filename, N, matrix):
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{matrix[i * N + j]} ")
            f.write("\n")

def generate_testcase_1():
    N = 100
    matrix = [round(random.uniform(-100.0, 100.0), 2) for _ in range(N * N)]
    write_matrix_to_file('testcases/1.in', N, matrix)

def generate_testcase_2():
    N = 3000
    matrix = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(N * N)]
    write_matrix_to_file('testcases/2.in', N, matrix)

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 小矩陣 ===")
    generate_testcase_1()
    
    print("=== 2.in: 大矩陣 ===")
    generate_testcase_2()
    
    for i in range(1, 3):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()