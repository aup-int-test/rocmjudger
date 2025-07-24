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
    M, N, d = 32, 64, 128
    Q = [round(random.uniform(-1.0, 1.0), 4) for _ in range(M * d)]
    K = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)]
    V = [round(random.uniform(-1.0, 1.0), 4) for _ in range(N * d)]
    write_attention_to_file('testcases/1.in', M, N, d, Q, K, V)

def generate_testcase_2():
    M, N, d = 256, 512, 128
    
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
    M, N, d = 1024, 2048, 256
    
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
    M, N, d = 1, 1, 1
    Q = [round(random.uniform(-1.0, 1.0), 4)]
    K = [round(random.uniform(-1.0, 1.0), 4)]
    V = [round(random.uniform(-1.0, 1.0), 4)]
    write_attention_to_file('testcases/4.in', M, N, d, Q, K, V)

def generate_testcase_5():
    M, N, d = 17, 33, 7
    Q = [round(random.uniform(-10.0, 10.0), 4) for _ in range(M * d)]
    K = [round(random.uniform(-10.0, 10.0), 4) for _ in range(N * d)]
    V = [round(random.uniform(-10.0, 10.0), 4) for _ in range(N * d)]
    write_attention_to_file('testcases/5.in', M, N, d, Q, K, V)

def generate_testcase_6():
    M, N, d = 1001, 2003, 257
    
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
    
    print("=== 1.in: 小測資 32x64x128 ===")
    generate_testcase_1()
    
    print("=== 2.in: 中測資 256x512x128 ===")
    generate_testcase_2()
    
    print("=== 3.in: 大測資 1024x2048x256 ===")
    generate_testcase_3()
    
    print("=== 4.in: Edge case 最小尺寸 1x1x1 ===")
    generate_testcase_4()
    
    print("=== 5.in: Edge case 不規則維度 17x33x7 ===")
    generate_testcase_5()
    
    print("=== 6.in: Edge case 不對齊block size 1001x2003x257 ===")
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()