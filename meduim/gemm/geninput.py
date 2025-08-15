import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_gemm_to_file(filename, M, N, K, alpha, beta, A, B, C=None):
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        f.write(f"{alpha} {beta}\n")
        
        for i in range(M):
            row_data = [A[i * K + j] for j in range(K)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(K):
            row_data = [B[i * N + j] for j in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        if C is None:
            C = [0.0] * (M * N)
        for i in range(M):
            row_data = [C[i * N + j] for j in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")

def generate_testcase_1():
    M, N, K = 2, 2, 2
    alpha, beta = 1.0, 0.0
    
    A = [1.0, 2.0, 3.0, 4.0]  # [[1,2], [3,4]]
    B = [5.0, 6.0, 7.0, 8.0]  # [[5,6], [7,8]]
    C = [0.0, 0.0, 0.0, 0.0]  # [[0,0], [0,0]]
    
    write_gemm_to_file('testcases/1.in', M, N, K, alpha, beta, A, B, C)

def generate_testcase_2():
    M, N, K = 256, 256, 128  
    alpha, beta = 2.0, 0.5
    
    random.seed(12345)
    A = [round(random.uniform(-1.0, 1.0), 4) for _ in range(M * K)]
    B = [round(random.uniform(-1.0, 1.0), 4) for _ in range(K * N)]
    C = [round(random.uniform(-0.5, 0.5), 4) for _ in range(M * N)]
    
    write_gemm_to_file('testcases/2.in', M, N, K, alpha, beta, A, B, C)

def generate_testcase_3():
    M, N, K = 128, 128, 64
    alpha, beta = 1.0, 1.0
    
    random.seed(23456)
    
    A = []
    for _ in range(M * K):
        if random.random() < 0.2:  # 20%非零
            A.append(round(random.uniform(-2.0, 2.0), 4))
        else:
            A.append(0.0)
    
    B = []
    for _ in range(K * N):
        if random.random() < 0.2:  # 20%非零
            B.append(round(random.uniform(-2.0, 2.0), 4))
        else:
            B.append(0.0)
    
    C = [round(random.uniform(-1.0, 1.0), 4) for _ in range(M * N)]
    
    write_gemm_to_file('testcases/3.in', M, N, K, alpha, beta, A, B, C)

def generate_testcase_4():
    M, N, K = 64, 64, 32
    alpha, beta = 3.0, 2.0
    
    A = [0.0] * (M * K)
    B = [0.0] * (K * N)
    
    random.seed(34567)
    C = [round(random.uniform(-5.0, 5.0), 4) for _ in range(M * N)]
    
    write_gemm_to_file('testcases/4.in', M, N, K, alpha, beta, A, B, C)

def generate_testcase_5():
    M, N, K = 1024, 1024, 512
    alpha, beta = 1.5, -0.5
    
    filename = 'testcases/5.in'
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        f.write(f"{alpha} {beta}\n")
        
        random.seed(45678)
        
        for i in range(M):
            row_data = [round(random.uniform(-3.0, 3.0), 6) for _ in range(K)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(K):
            row_data = [round(random.uniform(-3.0, 3.0), 6) for _ in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(M):
            row_data = [round(random.uniform(-1.0, 1.0), 6) for _ in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")

def generate_testcase_6():
    M, N, K = 2047, 2047, 1023 
    alpha, beta = -1.0, 2.5
    
    filename = 'testcases/6.in'
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        f.write(f"{alpha} {beta}\n")
        
        random.seed(56789)
        
        for i in range(M):
            row_data = [round(random.uniform(-10.0, 10.0), 6) for _ in range(K)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(K):
            row_data = [round(random.uniform(-10.0, 10.0), 6) for _ in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(M):
            row_data = [round(random.uniform(-5.0, 5.0), 6) for _ in range(N)]
            f.write(" ".join(map(str, row_data)) + "\n")

def main():
    
    create_testcases_dir()
    
    generate_testcase_1()
    
    generate_testcase_2()
    
    generate_testcase_3()
    
    generate_testcase_4()
    
    generate_testcase_5()
    
    generate_testcase_6()
    
    total_size = 0
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            total_size += file_size
            print(f"  {filename}: {file_size:,} bytes")
    

if __name__ == "__main__":
    main()