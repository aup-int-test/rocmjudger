import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_montecarlo_to_file(filename, a, b, n_samples, y_samples):
    with open(filename, 'w') as f:
        f.write(f"{a} {b} {n_samples}\n")
        f.write(" ".join(map(str, y_samples)) + "\n")

def generate_testcase_1():
    a, b = 0, 2
    n_samples = 8
    y_samples = [round(random.uniform(0.0, 5.0), 4) for _ in range(n_samples)]
    write_montecarlo_to_file('testcases/1.in', a, b, n_samples, y_samples)

def generate_testcase_2():
    a, b = 1, 1
    n_samples = 1
    y_samples = [round(random.uniform(0.0, 10.0), 4)]
    write_montecarlo_to_file('testcases/2.in', a, b, n_samples, y_samples)

def generate_testcase_3():
    a, b = 0, 10000000
    n_samples = 1000
    y_samples = [round(random.uniform(1e7, 1e9), 2) for _ in range(n_samples)]
    write_montecarlo_to_file('testcases/3.in', a, b, n_samples, y_samples)

def generate_testcase_4():
    a, b = -500, 1000
    n_samples = 100000
    y_samples = [round(random.uniform(0.0, 1000.0), 4) for _ in range(n_samples)]
    write_montecarlo_to_file('testcases/4.in', a, b, n_samples, y_samples)

def generate_testcase_5():
    a, b = -2000, -500
    n_samples = 100000
    y_samples = [round(random.uniform(0.0, 1000.0), 4) for _ in range(n_samples)]
    write_montecarlo_to_file('testcases/5.in', a, b, n_samples, y_samples)

def generate_testcase_6():
    a, b = 0, 1000
    n_samples = 100001
    y_samples = [round(random.uniform(0.0, 1000.0), 4) for _ in range(n_samples)]
    write_montecarlo_to_file('testcases/6.in', a, b, n_samples, y_samples)

def generate_testcase_7():
    a, b = 0, 100
    n_samples = 10000000
    
    filename = 'testcases/7.in'
    with open(filename, 'w') as f:
        f.write(f"{a} {b} {n_samples}\n")
        
        y_samples = [round(random.uniform(0.0, 200.0), 4) for _ in range(n_samples)]
        f.write(" ".join(map(str, y_samples)) + "\n")

def generate_testcase_8():
    a, b = 0, 1000
    n_samples = 100000000
    
    filename = 'testcases/8.in'
    with open(filename, 'w') as f:
        f.write(f"{a} {b} {n_samples}\n")
        
        y_samples = [round(random.uniform(0.0, 2000.0), 4) for _ in range(n_samples)]
        f.write(" ".join(map(str, y_samples)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 小測資 ===")
    generate_testcase_1()
    
    print("=== 2.in: N=1 ===")
    generate_testcase_2()
    
    print("=== 3.in: 數字大1e7～1e9 ===")
    generate_testcase_3()
    
    print("=== 4.in: 一正一負大測資 ===")
    generate_testcase_4()
    
    print("=== 5.in: 雙負大測資 ===")
    generate_testcase_5()
    
    print("=== 6.in: 正常測資 ===")
    generate_testcase_6()
    
    print("=== 7.in: 超大測資10000000 ===")
    generate_testcase_7()
    
    print("=== 8.in: 極大測資100000000 ===")
    generate_testcase_8()
    
    for i in range(1, 9):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()