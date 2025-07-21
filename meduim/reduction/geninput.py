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
    array_data = [round(random.uniform(-10.0, 10.0), 2) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 1000
    array_data = [round(random.uniform(-100.0, 100.0), 4) for _ in range(N)]
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 1000000
    
    filename = 'testcases/3.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_4():
    N = 10000000
    
    filename = 'testcases/4.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [round(random.uniform(-10000.0, 10000.0), 6) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 10 ===")
    generate_testcase_1()
    
    print("=== 2.in: 1000 正負數 ===")
    generate_testcase_2()
    
    print("=== 3.in: 1000000 隨機 ===")
    generate_testcase_3()
    
    print("=== 4.in: 10000000 隨機 ===")
    generate_testcase_4()
    
    for i in range(1, 5):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()