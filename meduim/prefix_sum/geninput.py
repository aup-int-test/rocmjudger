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
    N = 10
    array_data = [random.randint(-1000, 1000) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 1024
    array_data = [random.randint(-1000, 1000) for _ in range(N)]
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 10000000
    
    filename = 'testcases/3.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [random.randint(-1000, 1000) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_4():
    N = 1
    array_data = [random.randint(-1000, 1000)]
    write_array_to_file('testcases/4.in', N, array_data)

def generate_testcase_5():
    N = 1000
    array_data = [0] * N
    write_array_to_file('testcases/5.in', N, array_data)

def generate_testcase_6():
    N = 50000001
    
    filename = 'testcases/6.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [random.randint(-1000, 1000) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 小測資 10 ===")
    generate_testcase_1()
    
    print("=== 2.in: 中測資 1024 ===")
    generate_testcase_2()
    
    print("=== 3.in: 大測資 10000000 ===")
    generate_testcase_3()
    
    print("=== 4.in: Edge case 最小尺寸 1 ===")
    generate_testcase_4()
    
    print("=== 5.in: Edge case 全零陣列 1000 ===")
    generate_testcase_5()
    
    print("=== 6.in: 50000001 ===")
    generate_testcase_6()
    
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()