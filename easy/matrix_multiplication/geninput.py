import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_matrices_to_file(filename, M, N, K, matrix_a, matrix_b):
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        for i in range(M):
            for j in range(N):
                f.write(f"{matrix_a[i * N + j]} ")
            f.write("\n")
        for i in range(N):
            for j in range(K):
                f.write(f"{matrix_b[i * K + j]} ")
            f.write("\n")

def generate_testcase_1():
    M, N, K = 1, 1, 1
    matrix_a = [round(random.uniform(-10.0, 10.0), 2)]
    matrix_b = [round(random.uniform(-10.0, 10.0), 2)]
    write_matrices_to_file('testcases/1.in', M, N, K, matrix_a, matrix_b)

def generate_testcase_2():
    M, N, K = 100, 100, 100
    matrix_a = [round(random.uniform(-10.0, 10.0), 2) for _ in range(M * N)]
    matrix_b = [round(random.uniform(-10.0, 10.0), 2) for _ in range(N * K)]
    write_matrices_to_file('testcases/2.in', M, N, K, matrix_a, matrix_b)

def generate_testcase_3():
    M, N, K = 99, 87, 1000
    matrix_a = [round(random.uniform(-100.0, 100.0), 2) for _ in range(M * N)]
    matrix_b = [round(random.uniform(-100.0, 100.0), 2) for _ in range(N * K)]
    write_matrices_to_file('testcases/3.in', M, N, K, matrix_a, matrix_b)

def generate_testcase_4():
    M, N, K = 500, 300, 400
    matrix_a = [0.0] * (M * N)
    matrix_b = [0.0] * (N * K)
    write_matrices_to_file('testcases/4.in', M, N, K, matrix_a, matrix_b)

def generate_testcase_5():
    M, N, K = 1000, 1000, 1000
    matrix_a = []
    matrix_b = []
    
    for i in range(M * N):
        if random.random() < 0.05:
            matrix_a.append(round(random.uniform(-10.0, 10.0), 2))
        else:
            matrix_a.append(0.0)
    
    for i in range(N * K):
        if random.random() < 0.05:
            matrix_b.append(round(random.uniform(-10.0, 10.0), 2))
        else:
            matrix_b.append(0.0)
    
    write_matrices_to_file('testcases/5.in', M, N, K, matrix_a, matrix_b)

def generate_testcase_6():
    M, N, K = 1500, 2600, 1700
    matrix_a = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(M * N)]
    matrix_b = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(N * K)]
    write_matrices_to_file('testcases/6.in', M, N, K, matrix_a, matrix_b)

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: (1*1) * (1*1) ===")
    generate_testcase_1()
    
    print("=== 2.in: (100*100) * (100*100) ===")
    generate_testcase_2()
    
    print("=== 3.in: (99*87) * (87*1000) ===")
    generate_testcase_3()
    
    print("=== 4.in: 零矩陣 ===")
    generate_testcase_4()
    
    print("=== 5.in: 稀疏矩陣(1000*1000) * (1000*1000) ===")
    generate_testcase_5()
    
    print("=== 6.in: 較大隨機矩陣 ===")
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()