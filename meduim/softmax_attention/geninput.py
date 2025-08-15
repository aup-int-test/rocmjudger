import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_attention_to_file(filename, M, N, d, Q, K, V):
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {d}\n")
        for i in range(M):
            for j in range(d):
                f.write(f"{Q[i * d + j]} ")
            f.write("\n")
        for i in range(N):
            for j in range(d):
                f.write(f"{K[i * d + j]} ")
            f.write("\n")
        for i in range(N):
            for j in range(d):
                f.write(f"{V[i * d + j]} ")
            f.write("\n")

def generate_testcase_1():
    M, N, d = 1, 1, 1
    Q = [round(random.uniform(-1.0, 1.0), 4) for _ in range(M * d)]
    K = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)]
    V = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)]
    write_attention_to_file('testcases/1.in', M, N, d, Q, K, V)

def generate_testcase_2():
    M, N, d = 512, 512, 64
    
    filename = 'testcases/2.in'
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {d}\n")
        
        for i in range(M):
            row_data = [round(random.uniform(-2.0, 2.0), 4) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-2.0, 2.0), 4) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-2.0, 2.0), 4) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")

def generate_testcase_3():
    M, N, d = 1024, 1024, 128
    
    filename = 'testcases/3.in'
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {d}\n")
        
        for i in range(M):
            row_data = [round(random.uniform(-5.0, 5.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-5.0, 5.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-5.0, 5.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")

def generate_testcase_4():
    M, N, d = 2048, 2048, 256
    Q = [round(random.uniform(-1.0, 1.0), 4) for _ in range(M * d)]  
    K = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)] 
    V = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)]  
    write_attention_to_file('testcases/4.in', M, N, d, Q, K, V)

def generate_testcase_5():
    M, N, d = 4096, 4096, 512
    Q = [round(random.uniform(-10.0, 10.0), 4) for _ in range(M * d)]
    K = [round(random.uniform(-10.0, 10.0), 4) for _ in range(N * d)]
    V = [round(random.uniform(-10.0, 10.0), 4) for _ in range(N * d)]
    write_attention_to_file('testcases/5.in', M, N, d, Q, K, V)

def generate_testcase_6():
    M, N, d = 4095, 4095, 511
    
    filename = 'testcases/6.in'
    with open(filename, 'w') as f:
        f.write(f"{M} {N} {d}\n")
        
        for i in range(M):
            row_data = [round(random.uniform(-100.0, 100.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-100.0, 100.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")
        
        for i in range(N):
            row_data = [round(random.uniform(-100.0, 100.0), 6) for _ in range(d)]
            f.write(" ".join(map(str, row_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    generate_testcase_1()
    
    generate_testcase_2()
    
    generate_testcase_3()
    
    generate_testcase_4()
    
    generate_testcase_5()
    
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()